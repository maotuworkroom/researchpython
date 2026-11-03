#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""印章文字门禁 v2：tesseract OCR 版。

口径与旧启发式一致：只看圆内场景区域（排除上下 15% 色带与圆外）。
判定：pytesseract.image_to_data(-l jpn+eng --psm 11) 中 conf>=60 且
bbox 与场景掩膜重叠 >=50% 的词 => 命中；命中数>0 => 门禁不通过。

本文件在无 pytesseract 的环境下仍可导入（预处理/掩膜数值部分可本地复算），
只有真正跑 OCR 的 ocr_scene_text() 需要 tesseract（云端 runner 安装）。

CLI: python3 ocr_gate.py IMG.png [--json '{"psm":11,...}']
"""
import hashlib
import json
import sys

import numpy as np
import cv2

try:
    import pytesseract
    HAVE_TESS = True
except ImportError:
    pytesseract = None
    HAVE_TESS = False

try:
    import gate_lib as _G
    SIZE, C, R, BAND = _G.SIZE, _G.C, _G.R, _G.BAND
except ImportError:  # 独立运行（如云端只带本文件）
    SIZE = 2048
    C = SIZE // 2
    R = 1004
    BAND = 0.15

DEFAULTS = {
    "lang": "jpn+eng",
    "psm": 11,
    "oem": 1,              # 仅 LSTM 引擎（3=both 约慢一倍）
    "extra": "-c tessedit_do_invert=0",
    "conf_thr": 60.0,
    "scale": 1.0,          # <1 降采样 / 2 即"放大 2x"
    "binarize": None,      # None | "otsu" | "adaptive"
    "mask_out": True,      # 把场景掩膜外置白，减少干扰
    "crop": True,          # 裁到场景 bbox 再 OCR（掩膜外本就无有效内容）
    "denoise": 0,          # 中值模糊核（0=关）；抑制照片纹理碎斑，psm11 提速
    "min_overlap": 0.5,    # 命中 bbox 与掩膜重叠比例阈值
}


REGIONS = ("scene", "band_top", "band_bottom")


def band_top_y():
    return C - R + int(BAND * 2 * R)


def band_bottom_y():
    return SIZE - band_top_y()


def region_mask(region="scene"):
    """区域掩膜（三种口径，全部在 R-8 圆内）：
      scene       中段（去掉上下 15% 色带）——底图无字门禁
      band_top    上色带（品牌名嵌在这里）
      band_bottom 下色带（站名嵌在这里）
    """
    if region not in REGIONS:
        raise ValueError(f"region 必须是 {REGIONS}")
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    ins = (xs - C) ** 2 + (ys - C) ** 2 <= (R - 8) ** 2
    bt, bb = band_top_y(), band_bottom_y()
    if region == "scene":
        ins[:bt] = False
        ins[bb:] = False
    elif region == "band_top":
        ins[bt:] = False
    else:
        ins[:bb] = False
    return ins


def scene_mask():
    """兼容旧调用：场景区域（同 region_mask('scene')）。"""
    return region_mask("scene")


_REGION_CACHE = {}


def region_geom(region="scene"):
    """(bbox, mask) 按区域缓存。bbox=(x0,y0,x1,y1) 右开。"""
    if region not in _REGION_CACHE:
        ins = region_mask(region)
        ys, xs = np.where(ins)
        _REGION_CACHE[region] = (
            (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), ins)
    return _REGION_CACHE[region]


def _scene_bbox():
    return region_geom("scene")


def prepare_gray(rgba, scale=1.0, binarize=None, mask_out=True, crop=True,
                 denoise=0, region="scene"):
    """RGBA -> 灰度预处理图（tesseract 输入）。纯数值，本地可复算。
    返回 (gray, x0, y0)：x0/y0 为裁切偏移（命中坐标还原用）。"""
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    img = np.full((SIZE, SIZE, 3), 255, np.uint8)   # 透明处置白
    m = alpha > 127
    img[m] = rgb[m]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    (bx0, by0, bx1, by1), ins = region_geom(region)
    if mask_out:
        gray[~ins] = 255
    if crop:
        gray = gray[by0:by1, bx0:bx1]
        ins_c = ins[by0:by1, bx0:bx1]
    else:
        ins_c, bx0, by0 = ins, 0, 0
    if scale != 1.0:
        h, w = gray.shape
        interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=interp)
    if denoise and denoise > 1:
        gray = cv2.medianBlur(gray, int(denoise) | 1)
    if binarize == "otsu":
        _, gray = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 归一极性：掩膜内背景应为白（深色字）。若均值偏暗则整体取反。
        if scale != 1.0:
            ins_c = cv2.resize(ins_c.astype(np.uint8), (gray.shape[1], gray.shape[0]),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
        mu = float(gray[ins_c].mean()) if ins_c.any() else 255.0
        if mu < 127:
            gray = 255 - gray
    elif binarize == "adaptive":
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 51, 8)
    return gray, bx0, by0


def prep_fingerprint(rgba, scale=1.0, binarize=None, mask_out=True, crop=True,
                     denoise=0, region="scene"):
    """OCR 输入图的指纹（shape/crop/mean/std/otsu/sha1）。本机与云端共用同一函数，
    用于证明"喂给 tesseract 的是同一张图"（本机无 tesseract 时的逐项一致性依据）。"""
    gray, x0, y0 = prepare_gray(rgba, scale=scale, binarize=binarize, mask_out=mask_out,
                                crop=crop, denoise=denoise, region=region)
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return {"shape": list(gray.shape), "crop_xy": [int(x0), int(y0)],
            "mean": round(float(gray.mean()), 2), "std": round(float(gray.std()), 2),
            "otsu_thr": round(float(thr), 1),
            "prep_sha1": hashlib.sha1(gray.tobytes()).hexdigest()[:16]}


def geom_report():
    """区域几何常量（确定性，供本机/云端逐项比对）。"""
    return {"band_top_y": band_top_y(), "band_bottom_y": band_bottom_y(),
            "scene_bbox": list(region_geom("scene")[0]),
            "band_top_bbox": list(region_geom("band_top")[0]),
            "band_bottom_bbox": list(region_geom("band_bottom")[0])}


def ocr_scene_text(rgba, region="scene", **cfg):
    """OCR 文字门禁。返回 {n_hits, hits[], pass, config, ocr_sec}。

    region: scene（底图无字门禁）/ band_top（品牌名）/ band_bottom（站名）。
    """
    if not HAVE_TESS:
        raise RuntimeError("pytesseract 不可用（本机不装，云端 runner 才有）")
    import time
    c = dict(DEFAULTS)
    c.update({k: v for k, v in cfg.items() if v is not None})
    if "upscale" in cfg and cfg["upscale"]:  # 旧参数兼容
        c["scale"] = float(cfg["upscale"])
    gray, x0, y0 = prepare_gray(rgba, scale=c["scale"], binarize=c["binarize"],
                                mask_out=c["mask_out"], crop=c["crop"],
                                denoise=c.get("denoise", 0), region=region)
    _, ins = region_geom(region)
    prep_sha1 = hashlib.sha1(gray.tobytes()).hexdigest()[:16]
    t0 = time.time()
    d = pytesseract.image_to_data(gray, lang=c["lang"],
                                  config=f"--psm {c['psm']} --oem {c['oem']} {c['extra']}",
                                  output_type=pytesseract.Output.DICT)
    ocr_sec = time.time() - t0
    sc = c["scale"]
    hits = []
    weak = []   # conf 在 [conf_weak, conf_thr) 的词：用于区分"缺字"与"对比度不足"
    for i, txt in enumerate(d["text"]):
        try:
            conf = float(d["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        txt = (txt or "").strip()
        if not txt or conf < c.get("conf_weak", 35.0):
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        # 还原到 2048 原图坐标
        ox0 = int(x0 + x / sc)
        oy0 = int(y0 + y / sc)
        ox1 = min(SIZE, int(x0 + (x + w) / sc) + 1)
        oy1 = min(SIZE, int(y0 + (y + h) / sc) + 1)
        ox0, oy0 = max(0, ox0), max(0, oy0)
        sub = ins[oy0:oy1, ox0:ox1]
        if sub.size == 0:
            continue
        item = {"text": txt, "conf": round(conf, 1),
                "bbox": [ox0, oy0, ox1 - ox0, oy1 - oy0],
                "inside_frac": round(float(sub.mean()), 3)}
        if conf >= c["conf_thr"]:
            hits.append(item)
        else:
            weak.append(item)
    return {"n_hits": len(hits), "hits": hits, "n_weak": len(weak), "weak": weak,
            "pass": len(hits) == 0, "prep_sha1": prep_sha1,
            "config": {k: c[k] for k in DEFAULTS}, "ocr_sec": round(ocr_sec, 2)}


def main():
    from PIL import Image
    src = sys.argv[1]
    cfg = {}
    if "--json" in sys.argv:
        cfg = json.loads(sys.argv[sys.argv.index("--json") + 1])
    rgba = np.array(Image.open(src).convert("RGBA"))
    out = ocr_scene_text(rgba, **cfg)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
