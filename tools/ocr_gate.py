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
    "conf_thr": 60.0,
    "upscale": 1,          # 1 或 2（放大 2x）
    "binarize": None,      # None | "otsu" | "adaptive"
    "mask_out": True,      # 把场景掩膜外置白，减少干扰
    "min_overlap": 0.5,    # 命中 bbox 与掩膜重叠比例阈值
}


def scene_mask():
    """圆内、去掉上下 15% 色带后的场景区域（与 gate_lib.scene_mask 同口径）。"""
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    ins = (xs - C) ** 2 + (ys - C) ** 2 <= (R - 8) ** 2
    band_top = C - R + int(BAND * 2 * R)
    band_bot = SIZE - band_top
    ins[:band_top] = False
    ins[band_bot:] = False
    return ins


def prepare_gray(rgba, upscale=1, binarize=None, mask_out=True):
    """RGBA -> 灰度预处理图（tesseract 输入）。纯数值，本地可复算。"""
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    img = np.full((SIZE, SIZE, 3), 255, np.uint8)   # 透明处置白
    m = alpha > 127
    img[m] = rgb[m]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ins = scene_mask()
    if mask_out:
        gray[~ins] = 255
    if upscale != 1:
        gray = cv2.resize(gray, (SIZE * upscale, SIZE * upscale),
                          interpolation=cv2.INTER_CUBIC)
    if binarize == "otsu":
        _, gray = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 归一极性：掩膜内背景应为白（深色字）。若均值偏暗则整体取反。
        mu = float(gray[ins].mean()) if upscale == 1 else float(
            gray[np.kron(ins, np.ones((upscale, upscale), bool))].mean())
        if mu < 127:
            gray = 255 - gray
    elif binarize == "adaptive":
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 51, 8)
    return gray


def ocr_scene_text(rgba, **cfg):
    """OCR 文字门禁。返回 {n_hits, hits[], pass, config, ocr_sec}。"""
    if not HAVE_TESS:
        raise RuntimeError("pytesseract 不可用（本机不装，云端 runner 才有）")
    import time
    c = dict(DEFAULTS)
    c.update({k: v for k, v in cfg.items() if v is not None})
    gray = prepare_gray(rgba, upscale=c["upscale"], binarize=c["binarize"],
                        mask_out=c["mask_out"])
    ins = scene_mask()
    t0 = time.time()
    d = pytesseract.image_to_data(gray, lang=c["lang"],
                                  config=f"--psm {c['psm']}",
                                  output_type=pytesseract.Output.DICT)
    ocr_sec = time.time() - t0
    sc = c["upscale"]
    hits = []
    for i, txt in enumerate(d["text"]):
        try:
            conf = float(d["conf"][i])
        except (ValueError, TypeError):
            conf = -1.0
        txt = (txt or "").strip()
        if not txt or conf < c["conf_thr"]:
            continue
        x, y, w, h = d["left"][i], d["top"][i], d["width"][i], d["height"][i]
        x0, y0 = x // sc, y // sc
        x1 = min(SIZE, (x + w + sc - 1) // sc)
        y1 = min(SIZE, (y + h + sc - 1) // sc)
        sub = ins[y0:y1, x0:x1]
        if sub.size == 0:
            continue
        if float(sub.mean()) >= c["min_overlap"]:
            hits.append({"text": txt, "conf": round(conf, 1),
                         "bbox": [int(x0), int(y0), int(x1 - x0), int(y1 - y0)]})
    return {"n_hits": len(hits), "hits": hits, "pass": len(hits) == 0,
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
