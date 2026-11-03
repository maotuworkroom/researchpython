#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""印章文字门禁判定层：OCR 结果 -> 通过/失败 + 可定位的失败原因。

失败原因分类（5 类，命名写死）：
  missing_glyph          缺字：场景里没检出达标文字的字符覆盖不足
  glyph_substitution     字形替代：检出了文字，但与期望文本字符相似度低
  position_out_of_bounds 位置越界/溢出：命中框落在场景掩膜外（含色带/圆外）
  low_contrast           对比度不足：该区域笔画-背景的 RMS 对比度低于阈值
  mask_eats_glyph        遮罩吃掉字形：期望区域内的墨迹像素多数落在掩膜外
另有 unexpected_text：期望无字但检出了字（干净底图被写进字）。

阈值全部来自 ocr_gate_config.json，不在代码里散落。判定不可静默跳过：
evaluate() 必定返回 verdict∈{pass,fail}；异常路径由调用方记 error，不允许 return None。
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
    """读配置并附上 sha256 指纹（留痕：每次判定都带上 config_version + 指纹）。"""
    path = path or CFG_PATH
    raw = open(path, "rb").read()
    cfg = json.loads(raw.decode("utf-8"))
    for sec, keys in (("ocr", ("lang", "psm", "conf_hit")),
                      ("preprocess", ("scale", "binarize")),
                      ("base_clean", ("min_word_conf", "min_word_len")),
                      ("decision", ("pass_min_char_coverage",
                                    "substitution_max_char_mismatch",
                                    "low_contrast_rms_min",
                                    "band_overlap_max",
                                    "circle_outside_max",
                                    "out_of_bounds_inside_min"))):
        miss = [k for k in keys if k not in cfg.get(sec, {})]
        if miss:
            raise ValueError(f"配置缺字段 {sec}.{miss}")
    cfg["_sha256"] = hashlib.sha256(raw).hexdigest()[:16]
    cfg["_path"] = path
    return cfg


def ocr_cfg_from(cfg):
    """配置 -> ocr_gate.ocr_scene_text 的关键字参数。"""
    o, p = cfg["ocr"], cfg["preprocess"]
    return {"lang": o["lang"], "psm": o["psm"], "oem": o.get("oem", 1),
            "extra": o.get("extra", ""), "conf_thr": float(o["conf_hit"]),
            "conf_weak": float(o.get("conf_weak", 35)),
            "scale": float(p["scale"]), "binarize": p["binarize"],
            "denoise": int(p.get("denoise", 0)),
            "mask_out": bool(p["mask_out_scene"]),
            "crop": bool(p["crop_to_scene"]),
            "min_overlap": float(p["min_overlap"])}


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


def region_metrics(rgba, bbox, ink_dev_thr):
    """bbox 区域的对比度与"墨迹被遮罩吃掉"比例。bbox=(x0,y0,x1,y1)。"""
    _, ins = OG._scene_bbox()
    gray = gray_full(rgba)
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(OG.SIZE, x1), min(OG.SIZE, y1)
    g = gray[y0:y1, x0:x1]
    m = ins[y0:y1, x0:x1]
    if g.size == 0 or not m.any():
        return {"rms": 0.0, "ink_px": 0, "ink_outside_frac": 0.0, "region_px": 0}
    rms = float(g.std())
    med = float(np.median(g))
    ink = np.abs(g.astype(np.int16) - med) > ink_dev_thr
    ink_px = int(ink.sum())
    out = int((ink & ~m).sum())
    return {"rms": round(rms, 1), "ink_px": ink_px,
            "ink_outside_frac": round(out / ink_px, 3) if ink_px else 0.0,
            "region_px": int(g.size)}


def _hits_bbox(hits):
    xs0 = min(h["bbox"][0] for h in hits)
    ys0 = min(h["bbox"][1] for h in hits)
    xs1 = max(h["bbox"][0] + h["bbox"][2] for h in hits)
    ys1 = max(h["bbox"][1] + h["bbox"][3] for h in hits)
    return (xs0, ys0, xs1, ys1)


def rect_geometry(bbox):
    """字形矩形相对"场景掩膜"的解析几何（不依赖 OCR，纯坐标判定）。

    返回:
      band_frac   矩形落在上下 15% 色带内的像素比例 -> 遮罩吃掉字形
      outside_frac 矩形落在圆外的像素比例        -> 位置越界/溢出
    """
    _, ins = OG._scene_bbox()
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(OG.SIZE, x1), min(OG.SIZE, y1)
    if x1 <= x0 or y1 <= y0:
        return {"band_frac": 1.0, "outside_frac": 1.0, "rect_px": 0}
    band_top = OG.C - OG.R + int(OG.BAND * 2 * OG.R)
    band_bot = OG.SIZE - band_top
    ys, xs = np.mgrid[y0:y1, x0:x1]
    in_circle = (xs - OG.C) ** 2 + (ys - OG.C) ** 2 <= (OG.R - 8) ** 2
    in_band = (ys < band_top) | (ys >= band_bot)
    px = int(in_circle.size)
    return {"band_frac": round(float((in_band & in_circle).sum()) / px, 3),
            "outside_frac": round(float((~in_circle).sum()) / px, 3),
            "rect_px": px}


_WORD_RE = re.compile(r"^[0-9A-Za-z\u3040-\u30ff\u3400-\u9fff\uff66-\uff9f]+$")


def filter_hits(hits, f):
    """词有效性过滤：把照片纹理产生的符号噪声（=, >, |, ¢, 单字母）剔掉。

    参数全部来自配置 base_clean 段：
      min_word_conf     conf 下限（噪声词置信通常 60~72，合成字 85+）
      min_word_len      最短词长（站名 >=3 字）
      min_glyph_h_px    命中原图字高下限（排除纹理碎斑）
      min_width_frac    命中宽度占场景宽比例下限（排除小碎块）
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
        if h["bbox"][3] < int(f.get("min_glyph_h_px", 0)):
            continue
        if f.get("scene_w"):
            frac = h["bbox"][2] / float(f["scene_w"])
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


def classify(ocr, cfg, expected_text="", expected_bbox=None, rgba=None):
    """纯判定：输入 ocr 结果 + 配置 + 期望文本/区域，输出 verdict/reason/metrics。"""
    d = cfg["decision"]
    hits, weak = ocr["hits"], ocr.get("weak", [])
    joined = "".join(h["text"] for h in hits)
    recall = char_coverage(expected_text, joined) if expected_text else 1.0
    precision = char_coverage(joined, expected_text) if expected_text else 1.0
    metrics = {
        "n_hits": len(hits), "n_weak": len(weak),
        "char_recall": round(recall, 3),
        "char_precision": round(precision, 3),
        "mismatch": round(1.0 - precision, 3) if expected_text else 0.0,
        "similarity": round(max((similarity(expected_text, h["text"]) for h in hits),
                                default=0.0), 3) if expected_text else 1.0,
        "max_inside_frac": round(max((h["inside_frac"] for h in hits), default=1.0), 3),
        "prep_sha1": ocr.get("prep_sha1"),
    }
    if expected_bbox:
        metrics.update(rect_geometry(expected_bbox))
    if rgba is not None:
        probe = expected_bbox or (_hits_bbox(hits) if hits else None)
        if probe:
            metrics.update(region_metrics(rgba, probe, d["ink_dev_thr"]))

    # 1) 字形矩形越出圆外 -> 位置越界/溢出
    if metrics.get("outside_frac", 0) > d["circle_outside_max"]:
        return "fail", "position_out_of_bounds", metrics
    # 2) 字形矩形压进上下色带 -> 遮罩吃掉字形
    if metrics.get("band_frac", 0) > d["band_overlap_max"]:
        return "fail", "mask_eats_glyph", metrics
    # 3) 无命中
    if not hits:
        if expected_text:
            if metrics.get("rms", 255) < d["low_contrast_rms_min"] or weak:
                return "fail", "low_contrast", metrics
            return "fail", "missing_glyph", metrics
        return "pass", None, metrics
    # 4) 期望无字却检出 -> 干净底图被写进了字
    if not expected_text:
        return "fail", "unexpected_text", metrics
    # 5) 字符召回不足 -> 缺字；召回够但检出的字大多不是期望字 -> 字形替代
    if metrics["char_recall"] < d["pass_min_char_coverage"]:
        return "fail", "missing_glyph", metrics
    if metrics["mismatch"] > d["substitution_max_char_mismatch"]:
        return "fail", "glyph_substitution", metrics
    if metrics["max_inside_frac"] < d["out_of_bounds_inside_min"]:
        return "fail", "position_out_of_bounds", metrics
    return "pass", None, metrics


# region_metrics 需要 rgba 作上下文，由 evaluate 显式传入
def evaluate(rgba, cfg, expected_text="", expected_bbox=None, ocr_result=None):
    """端到端：跑 OCR + 分类。返回 {verdict, reason, metrics, ocr_sec, config_version}。"""
    ocr = ocr_result or OG.ocr_scene_text(rgba, **ocr_cfg_from(cfg))
    verdict, reason, metrics = classify(ocr, cfg, expected_text, expected_bbox, rgba)
    return {"verdict": verdict, "reason": reason,
            "metrics": metrics, "ocr_sec": ocr["ocr_sec"],
            "hits": ocr["hits"][:8],
            "config_version": cfg["config_version"],
            "config_sha256": cfg["_sha256"]}


def main():
    from PIL import Image
    if len(sys.argv) < 2:
        sys.exit("用法: python3 gate_decision.py IMG.png [期望文本] [--no-ocr]")
    rgba = np.array(Image.open(sys.argv[1]).convert("RGBA"))
    expected = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else ""
    cfg = load_config()
    if "--no-ocr" in sys.argv:      # 只复算几何/对比度，不碰 tesseract
        out = {"metrics": region_metrics(rgba, OG._scene_bbox()[0], 40.0),
               "config_version": cfg["config_version"], "config_sha256": cfg["_sha256"]}
    else:
        out = evaluate(rgba, cfg, expected)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
