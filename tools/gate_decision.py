#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""印章文字门禁判定层：OCR 结果 -> 通过/失败 + 可定位的失败原因。

两道门禁、共 6 类原因（命名写死，写进结果与计数）：
  missing_glyph          缺字    期望有字，区域里墨迹/字符覆盖不足
  glyph_substitution     字形替代 检出文字，但与期望文本字符不匹配（错字）
  position_out_of_bounds 位置越界 墨迹/命中越出圆形边界（溢出）
  low_contrast           对比度不足 有淡墨迹但 RMS/置信度不足，OCR 读不出
  mask_eats_glyph        遮罩吃掉字形 墨迹跨出色带边界，被门禁掩膜裁掉
  unexpected_text        期望无字却检出字（AI 底图场景被画进了文字）

阈值全部来自 ocr_gate_config.json（decision / base_clean 两段），代码里不散落魔数；
每次判定都带 config_version + config_sha256 留痕。判定不可静默跳过：evaluate()
必定返回 verdict∈{pass,fail}，异常由调用方记 error。
"""
import hashlib
import json
import os
import re
import sys

import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocr_gate as OG  # noqa: E402

CFG_PATH = os.path.join(HERE, "ocr_gate_config.json")
REASONS = ("missing_glyph", "glyph_substitution", "position_out_of_bounds",
           "low_contrast", "mask_eats_glyph", "unexpected_text")


def load_config(path=None):
    """读配置并附 sha256 指纹（留痕：每次判定都带 config_version + 指纹）。"""
    path = path or CFG_PATH
    raw = open(path, "rb").read()
    cfg = json.loads(raw.decode("utf-8"))
    for sec, keys in (("ocr", ("lang", "psm", "conf_hit")),
                      ("preprocess", ("scale", "binarize")),
                      ("base_clean", ("min_word_conf", "min_word_len")),
                      ("decision", ("pass_min_char_coverage",
                                    "substitution_max_char_mismatch",
                                    "circle_outside_max",
                                    "ink_outside_max",
                                    "ink_dev_thr",
                                    "text_contrast_min",
                                    "glyph_cov_min",
                                    "glyph_ink_min"))):
        miss = [k for k in keys if k not in cfg.get(sec, {})]
        if miss:
            raise ValueError(f"配置缺字段 {sec}.{miss}")
    cfg["_sha256"] = hashlib.sha256(raw).hexdigest()[:16]
    cfg["_path"] = path
    return cfg


def ocr_cfg_from(cfg, region="scene"):
    """配置 -> ocr_gate.ocr_scene_text 的关键字参数（色带按 band_* 覆盖）。"""
    o, p = cfg["ocr"], cfg["preprocess"]
    band = region.startswith("band_")
    return {"lang": o["lang"], "psm": int(p.get("band_psm", o["psm"]) if band else o["psm"]),
            "oem": o.get("oem", 1),
            "extra": o.get("extra", ""), "conf_thr": float(o["conf_hit"]),
            "conf_weak": float(o.get("conf_weak", 35)),
            "scale": float(p.get("band_scale", p["scale"]) if band else p["scale"]),
            "binarize": p["binarize"],
            "denoise": int(p.get("band_denoise", 0) if band else p.get("denoise", 0)),
            "mask_out": bool(p.get("band_mask_out", p["mask_out_scene"])
                             if band else p["mask_out_scene"]),
            "crop": bool(p["crop_to_scene"]),
            "min_overlap": float(p["min_overlap"])}


def prep_cfg_from(cfg, region="scene"):
    """配置 -> ocr_gate.prep_fingerprint 的关键字参数（预处理子集）。"""
    c = ocr_cfg_from(cfg, region)
    return {k: c[k] for k in ("scale", "binarize", "mask_out", "crop", "denoise")}


def levenshtein(a, b):
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def char_coverage(expected, observed):
    """期望字符被观测文本覆盖的比例（多重集口径，忽略顺序）。"""
    if not expected:
        return 1.0
    pool = {}
    for ch in observed:
        pool[ch] = pool.get(ch, 0) + 1
    got = 0
    for ch in expected:
        if pool.get(ch, 0) > 0:
            pool[ch] -= 1
            got += 1
    return got / len(expected)


def similarity(expected, observed):
    if not expected:
        return 1.0
    return 1.0 - levenshtein(expected, observed) / max(len(expected), len(observed), 1)


def gray_full(rgba):
    """与 OCR 输入同口径的灰度（裁切/缩放前），用于几何与对比度度量。"""
    rgb, alpha = rgba[:, :, :3], rgba[:, :, 3]
    img = np.full((OG.SIZE, OG.SIZE, 3), 255, np.uint8)
    m = alpha > 127
    img[m] = rgb[m]
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


_CIRC = None


def circle_mask():
    """(R-8) 圆盘掩膜（缓存）。"""
    global _CIRC
    if _CIRC is None:
        ys, xs = np.mgrid[0:OG.SIZE, 0:OG.SIZE]
        _CIRC = (xs - OG.C) ** 2 + (ys - OG.C) ** 2 <= (OG.R - 8) ** 2
    return _CIRC


def _clip(b):
    x0, y0, x1, y1 = [int(v) for v in b]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(OG.SIZE, x1), min(OG.SIZE, y1)
    return x0, y0, x1, y1


def expand_box(box_xywh, frac=0.15):
    """[x,y,w,h] -> (x0,y0,x1,y1)：按宽高比例外扩，供 OCR 紧贴裁切（色带用）。

    以 pipeline.text_box 的同字体复算字框为基准，外扩 frac 后作为 tesseract 的输入窗口：
    既去掉满幅留白（psm 7 单行模式在大片空白上会放弃识别），又给字缘留出落笔余量。
    """
    if not box_xywh:
        return None
    x, y, w, h = [int(v) for v in box_xywh]
    px, py = int(round(w * frac)), int(round(h * frac))
    return _clip((x - px, y - py, x + w + px, y + h + py))


def rect_geometry(bbox, region="scene"):
    """字形矩形相对"圆盘"与"区域掩膜"的解析几何（纯坐标，不依赖 OCR）。"""
    x0, y0, x1, y1 = _clip(bbox)
    if x1 <= x0 or y1 <= y0:
        return {"circle_outside_frac": 1.0, "region_outside_frac": 1.0, "rect_px": 0}
    circ = circle_mask()[y0:y1, x0:x1]
    reg = OG.region_geom(region)[1][y0:y1, x0:x1]
    px = int(circ.size)
    return {"circle_outside_frac": round(float((~circ).sum()) / px, 3),
            "region_outside_frac": round(float((~reg & circ).sum()) / px, 3),
            "rect_px": px}


def ink_metrics(rgba, region="band_bottom", region_bbox=None, ink_dev_thr=40.0,
                ink_dev_soft=12.0, margin=24, min_comp_px=8):
    """色带墨迹度量（判"遮罩吃掉字形 / 位置越界 / 对比度不足"）。

    做法：以区域内灰度中位数为背景色，在"色带 + 朝场景一侧外扩 margin 像素"的窗口里取
    墨迹连通域，只保留"与区域本身有交叠"的连通域（照片纹理不与色带文字连通 -> 被剔除）。
    返回:
      rms                 区域内灰度标准差（对比度）
      ink_px / ink_soft_px 硬/软阈值下的墨迹像素数（软阈值用于识别"淡到读不出"）
      ink_region_outside_frac 墨迹落在区域外（色带外）的比例 -> 遮罩吃掉字形
      ink_circle_outside_frac 墨迹落在圆外的比例           -> 位置越界
      ink_bbox            墨迹包围盒（原图坐标）
      window              统计窗口
    """
    gray = gray_full(rgba)
    alpha_ok = rgba[:, :, 3] > 127        # 印章本体（圆内 + 落在色带外的字）才可能是墨迹
    reg = OG.region_geom(region)[1]
    circ = circle_mask()
    box = _clip(region_bbox or OG.region_geom(region)[0])
    wx0, wy0, wx1, wy1 = box
    if region == "band_top":
        wy1 = min(OG.SIZE, wy1 + margin)
    elif region == "band_bottom":
        wy0 = max(0, wy0 - margin)
    rms = float(gray[reg].std()) if reg.any() else 0.0
    med = float(np.median(gray[reg])) if reg.any() else 255.0
    g = gray[wy0:wy1, wx0:wx1]
    ok = alpha_ok[wy0:wy1, wx0:wx1]
    hard = (np.abs(g.astype(np.int16) - med) > ink_dev_thr) & ok
    soft = (np.abs(g.astype(np.int16) - med) > ink_dev_soft) & ok
    n, labels, stats, _ = cv2.connectedComponentsWithStats(hard.astype(np.uint8), 8)
    reg_sub = reg[wy0:wy1, wx0:wx1]
    circ_sub = circ[wy0:wy1, wx0:wx1]
    keep = np.zeros(n, bool)
    if n > 1 and reg_sub.any():
        ids = np.unique(labels[reg_sub])
        for i in ids:
            if i and stats[i][4] >= min_comp_px:
                keep[i] = True
    kept = keep[labels] if n > 1 else np.zeros_like(reg_sub)
    ink_px = int(kept.sum())
    out_region = int((kept & ~reg_sub).sum())
    out_circle = int((kept & ~circ_sub).sum())
    soft_px = int((soft & reg_sub).sum())
    bbox = None
    if ink_px:
        ys, xs = np.where(kept)
        bbox = (int(wx0 + xs.min()), int(wy0 + ys.min()),
                int(wx0 + xs.max()) + 1, int(wy0 + ys.max()) + 1)
    geo = rect_geometry(bbox, region) if bbox else {"circle_outside_frac": 0.0,
                                                   "region_outside_frac": 0.0,
                                                   "rect_px": 0}
    return {"rms": round(rms, 1), "ink_px": ink_px, "ink_soft_px": soft_px,
            "ink_region_outside_frac": round(out_region / ink_px, 3) if ink_px else 0.0,
            "ink_circle_outside_frac": round(out_circle / ink_px, 3) if ink_px else 0.0,
            "ink_bbox": list(bbox) if bbox else None,
            "ink_bbox_geometry": geo,
            "window": [wx0, wy0, wx1, wy1]}


_WORD_RE = re.compile(r"^[0-9A-Za-z\u3040-\u30ff\u3400-\u9fff\uff66-\uff9f]+$")


def region_rms(rgba, region="scene"):
    """区域灰度标准差（对比度指标，纯数值）。"""
    reg = OG.region_geom(region)[1]
    return round(float(gray_full(rgba)[reg].std()), 1)


def text_contrast(rgba, bbox_xywh, soft=12.0):
    """期望字框内的前景/背景灰度差（确定性像素度量，用于判"对比度不足"）。

    bbox_xywh = [x,y,w,h]（与 pipeline.text_box 同口径）。
    以框内灰度中位数为背景，取"偏离背景 > soft"的像素为墨迹，返回墨迹中位灰度与
    背景之差。不依赖 OCR 是否读得出，故不受 tesseract 偶发成败影响。
    """
    if not bbox_xywh:
        return 0.0
    x, y, w, h = [int(v) for v in bbox_xywh]
    x0, y0, x1, y1 = _clip((x, y, x + w, y + h))
    gray = gray_full(rgba)
    ok = rgba[:, :, 3] > 127
    g = gray[y0:y1, x0:x1].astype(np.int16)
    m = ok[y0:y1, x0:x1]
    if g.size == 0 or not m.any():
        return 0.0
    vals = g[m]
    bg = float(np.median(vals))
    ink = vals[np.abs(vals - bg) > soft]
    if ink.size < 32:
        return 0.0
    return round(float(abs(np.median(ink) - bg)), 1)


def ink_mask(rgba, region, dev=40.0):
    """区域内的实际墨迹掩膜（以区域内灰度为背景，偏离 > dev 且不透明）。"""
    gray = gray_full(rgba)
    reg = OG.region_geom(region)[1]
    bg = float(np.median(gray[reg])) if reg.any() else 255.0
    return (np.abs(gray.astype(np.int16) - bg) > dev) & (rgba[:, :, 3] > 127)


def glyph_template(rgba, text, top=False, region="band_bottom", dev=40.0):
    """逐字模板比对：期望字迹（同字体同拟合复算） vs 实际墨迹（确定性，不依赖 OCR）。

    每个期望字给出 coverage（期望笔画被实际墨迹覆盖的比例）与 precision
    （该字框内实际墨迹落在期望笔画上的比例）：
      缺字    -> 该字 ink_px≈0，coverage≈0
      错字    -> 该字 ink_px 正常，但覆盖/精确都低（字形不在期望笔画上）
    """
    import pipeline as P
    exp, cells = P.text_mask(text, top)
    act = ink_mask(rgba, region, dev)
    per = []
    for ch, (x, y, w, h) in zip(text, cells):
        x0, y0, x1, y1 = _clip((x, y, x + w, y + h))
        e = exp[y0:y1, x0:x1]
        a = act[y0:y1, x0:x1]
        ep, ap = int(e.sum()), int(a.sum())
        ov = int((e & a).sum())
        per.append({"ch": ch, "box": [x0, y0, x1 - x0, y1 - y0], "exp_px": ep,
                    "ink_px": ap,
                    "coverage": round(ov / ep, 3) if ep else 1.0,
                    "precision": round(ov / ap, 3) if ap else 0.0})
    return {"cells": per,
            "min_coverage": min([c["coverage"] for c in per] or [1.0]),
            "min_precision": min([c["precision"] for c in per] or [1.0]),
            "ink_px": sum(c["ink_px"] for c in per)}


def filter_hits(hits, f):
    """词有效性过滤：把照片纹理产生的符号噪声（=, >, |, ¢, 单字母）剔掉。

    参数全部来自配置 base_clean 段：
      min_word_conf     conf 下限（噪声词置信通常 60~72，合成字 85+）
      min_word_len      最短词长（站名 >=3 字）
      min_glyph_h_px    命中原图字高下限（排除纹理碎斑）
      min_width_frac    命中宽度占场景宽比例下限（排除小碎块）
      min_aspect        命中框宽高比下限（真实站名是一行字，宽/高 >= 2.5）
      scene_w           场景宽（原图 px）
    """
    out = []
    for h in hits:
        t = (h.get("text") or "").strip()
        if len(t) < int(f.get("min_word_len", 2)):
            continue
        if not _WORD_RE.match(t):
            continue
        if float(h.get("conf", 0)) < float(f.get("min_word_conf", 60)):
            continue
        w, hh = h["bbox"][2], h["bbox"][3]
        if hh < int(f.get("min_glyph_h_px", 0)):
            continue
        if f.get("min_aspect") and hh > 0 and (w / float(hh)) < float(f["min_aspect"]):
            continue
        if f.get("scene_w"):
            frac = w / float(f["scene_w"])
            if frac < float(f.get("min_width_frac", 0)):
                continue
        out.append(h)
    return out


def base_clean_verdict(hits, f):
    """底图无字门禁：过滤后仍有词 => unexpected_text(fail)，否则 pass。"""
    kept = filter_hits(hits, f)
    if kept:
        return "fail", "unexpected_text", [h["text"] for h in kept[:6]]
    return "pass", None, []


def classify(ocr, cfg, expected_text="", region="scene", rgba=None, expected_bbox=None):
    """纯判定：OCR 结果 + 配置 + 期望文本/区域 -> verdict/reason/metrics。

    expected_bbox: 嵌字"应有"的墨迹包围盒 [x,y,w,h]（pipeline.text_box 复算）。
    给了它就能用"墨迹宽度比"判缺字，不依赖 OCR 的逐字召回（tesseract 偶发丢字）。
    """
    d = cfg["decision"]
    hits, weak = ocr["hits"], ocr.get("weak", [])
    joined = "".join(h["text"] for h in hits)
    recall = char_coverage(expected_text, joined) if expected_text else 1.0
    precision = char_coverage(joined, expected_text) if expected_text else 1.0
    metrics = {
        "region": region,
        "n_hits": len(hits), "n_weak": len(weak),
        "char_recall": round(recall, 3),
        "char_precision": round(precision, 3),
        "mismatch": round(1.0 - precision, 3) if expected_text else 0.0,
        "similarity": round(max((similarity(expected_text, h["text"]) for h in hits),
                                default=0.0), 3) if expected_text else 1.0,
        "max_inside_frac": round(max((h["inside_frac"] for h in hits), default=1.0), 3),
        "prep_sha1": ocr.get("prep_sha1"),
    }
    if rgba is not None and expected_text:
        metrics.update(ink_metrics(
            rgba, region, ink_dev_thr=float(d["ink_dev_thr"]),
            ink_dev_soft=float(d.get("ink_dev_soft", 12.0)),
            margin=int(d.get("ink_margin_px", 24))))
    if rgba is not None and expected_text and region.startswith("band_"):
        # 逐字模板比对：缺字/错字的确定性判据（不依赖 tesseract 偶发整行失败）
        tm = glyph_template(rgba, expected_text, top=(region == "band_top"),
                            region=region, dev=float(d["ink_dev_thr"]))
        metrics["template"] = {k: tm[k] for k in ("min_coverage", "min_precision",
                                                  "ink_px")}
        metrics["template_cells"] = tm["cells"]
    if rgba is not None and expected_text and expected_bbox:
        metrics["text_contrast"] = text_contrast(
            rgba, expected_bbox, soft=float(d.get("ink_dev_soft", 12.0)))
    if expected_bbox and metrics.get("ink_bbox"):
        bx0, by0, bx1, by1 = metrics["ink_bbox"]
        ew = max(float(expected_bbox[2]), 1.0)
        eh = max(float(expected_bbox[3]), 1.0)
        metrics["expected_bbox"] = list(expected_bbox)
        metrics["width_ratio"] = round((bx1 - bx0) / ew, 3)
        metrics["height_ratio"] = round((by1 - by0) / eh, 3)

    # 1) 墨迹越出圆盘 -> 位置越界/溢出
    if metrics.get("ink_circle_outside_frac", 0.0) > d["circle_outside_max"]:
        return "fail", "position_out_of_bounds", metrics
    # 2) 墨迹跨出色带边界（被门禁掩膜裁掉）-> 遮罩吃掉字形
    if metrics.get("ink_region_outside_frac", 0.0) > d["ink_outside_max"]:
        return "fail", "mask_eats_glyph", metrics
    # 3) 墨迹宽度明显短于应有宽度 -> 缺字（几何判据，先于任何字符级判据）
    if metrics.get("width_ratio") is not None and \
            metrics["width_ratio"] < d["min_width_ratio"]:
        return "fail", "missing_glyph", metrics
    # 4) 字框内前景/背景灰度差不足 -> 对比度不足（确定性像素度量）
    if metrics.get("text_contrast") is not None and \
            metrics["text_contrast"] < d["text_contrast_min"]:
        return "fail", "low_contrast", metrics
    # 5) 逐字模板：某字笔画缺失/被换成别的字形 -> 缺字 / 字形替代
    if metrics.get("template_cells"):
        for c in metrics["template_cells"]:
            if c["coverage"] < d["glyph_cov_min"]:
                if c["ink_px"] < int(d["glyph_ink_min"]):
                    return "fail", "missing_glyph", metrics
                return "fail", "glyph_substitution", metrics
    # 6) 场景区域（期望无字）：OCR 读到有效词 -> 底图被画进字（该原因只上报，见 base_clean）
    if hits and not expected_text:
        return "fail", "unexpected_text", metrics
    # 7) 色带：OCR 读字作为交叉校验，只记录不阻断。
    #    实测 psm7 在合成 CJK 字上有整行读不出（3/6）与误读（3/6）的情况，
    #    若把它当硬判据会把干净样本判失败，故 5 类硬原因全部由确定性度量给出。
    return "pass", None, metrics


def evaluate(rgba, cfg, expected_text="", region="scene", ocr_result=None):
    """端到端：跑 OCR + 分类。返回 {verdict, reason, metrics, ocr_sec, config_version}。"""
    ocr = ocr_result or OG.ocr_scene_text(rgba, region=region,
                                          **ocr_cfg_from(cfg, region))
    verdict, reason, metrics = classify(ocr, cfg, expected_text, region, rgba)
    return {"verdict": verdict, "reason": reason,
            "metrics": metrics, "ocr_sec": ocr["ocr_sec"],
            "hits": ocr["hits"][:8],
            "config_version": cfg["config_version"],
            "config_sha256": cfg["_sha256"]}


def main():
    from PIL import Image
    usage = "用法: python3 gate_decision.py IMG.png [期望文本] [--region band_bottom]"
    if len(sys.argv) < 2:
        sys.exit(usage)
    rgba = np.array(Image.open(sys.argv[1]).convert("RGBA"))
    expected = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else ""
    region = "scene"
    if "--region" in sys.argv:
        region = sys.argv[sys.argv.index("--region") + 1]
    cfg = load_config()
    out = evaluate(rgba, cfg, expected, region)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
