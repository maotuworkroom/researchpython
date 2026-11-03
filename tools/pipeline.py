#!/usr/bin/env python3
# 单站流水线：原图 -> 圆形度门禁 -> 归一化底图 -> 文字/裁断门禁 -> 嵌字 -> 复检
# 用法: nice -n 19 python3 pipeline.py RAW OUT_BASE OUT_FINAL STATION BRAND [--json OUT.json]
import json
import sys
import time

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

import gate_lib as G

SIZE, C, R, BAND = G.SIZE, G.C, G.R, G.BAND
FONT_CANDIDATES = [
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 1),
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
    ("/System/Library/Fonts/PingFang.ttc", 1),
]
GATE_ASPECT = 0.97      # 椭圆比阈值
GATE_TEXT_ROWS = 0      # 允许的场景文字行数
GATE_TRUNC = 0          # 允许的主体裁断数


def normalize(rgb):
    c, m = G.content_mask_rgb(rgb)
    if c is None:
        raise RuntimeError("no content mask")
    (cx, cy), rad = cv2.minEnclosingCircle(c)
    x, y, w, h = cv2.boundingRect(c)
    crop = rgb[y:y + h, x:x + w]
    inner = cv2.resize(crop, (2 * R, 2 * R), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((SIZE, SIZE, 3), np.uint8)
    canvas[C - R:C + R, C - R:C + R] = inner
    big = np.zeros((SIZE * 4, SIZE * 4), np.uint8)
    cv2.circle(big, (C * 4, C * 4), R * 4, 255, -1, cv2.LINE_AA)
    alpha = cv2.resize(big, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    return np.dstack([canvas, alpha])


def band_color(rgba, top):
    y0 = C - R + int(0.03 * 2 * R)
    y1 = C - R + int(0.10 * 2 * R)
    if not top:
        y0, y1 = SIZE - y1, SIZE - y0
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    inside = (xs - C) ** 2 + (ys - C) ** 2 <= (R - 6) ** 2
    sel = inside.copy()
    sel[:y0] = False
    sel[y1:] = False
    px = rgba[:, :, :3][sel]
    return np.median(px, axis=0).astype(np.uint8)


def flatten_bands(rgba):
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    inside = (xs - C) ** 2 + (ys - C) ** 2 <= R ** 2
    for top in (True, False):
        col = band_color(rgba, top)
        y0 = C - R
        y1 = C - R + int(BAND * 2 * R)
        if not top:
            y0, y1 = SIZE - y1, SIZE - y0
        region = inside.copy()
        region[:y0] = False
        region[y1:] = False
        rgba[:, :, :3][region] = col
    return rgba


_FONT_CACHE = {}


def pick_font(size):
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    for path, idx in FONT_CANDIDATES:
        try:
            f = ImageFont.truetype(path, size, index=idx)
            _FONT_CACHE[size] = f
            return f
        except Exception:
            continue
    raise RuntimeError("no CJK font")


def fit_font(text, max_w, max_h, start):
    """二分可用最大字号（逐档 load 大 TTC 在 Linux runner 上很慢，见 _FONT_CACHE）。"""
    lo, hi, best = 12, start, None
    while lo <= hi:
        mid = (lo + hi) // 2
        f = pick_font(mid)
        l, t, r, b = f.getbbox(text)
        if (r - l) <= max_w and (b - t) <= max_h:
            best = f
            lo = mid + 1
        else:
            hi = mid - 1
    return best or pick_font(12)


def emboss(rgba, station, brand):
    img = Image.fromarray(rgba)
    d = ImageDraw.Draw(img)
    band_h = int(BAND * 2 * R)
    for top, text in ((True, brand), (False, station)):
        col = band_color(rgba, top)
        lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
        fg = (31, 36, 48, 255) if lum > 140 else (255, 255, 255, 255)
        y_mid = (C - R + band_h // 2) if top else (SIZE - (C - R) - band_h // 2)
        dy = abs(C - y_mid)
        chord = 2 * int(np.sqrt(max(R * R - dy * dy, 0)))
        f = fit_font(text, int(chord * 0.78), int(band_h * 0.62),
                     int(band_h * 0.40) if top else int(band_h * 0.66))
        l, t, r, b = f.getbbox(text)
        w, h = r - l, b - t
        d.text((C - w / 2 - l, y_mid - h / 2 - t), text, font=f, fill=fg)
    return np.array(img)


def band_text_check(rgba):
    """复检：色带内是否真有嵌字（连通域计数）。"""
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    inside = (xs - C) ** 2 + (ys - C) ** 2 <= (R - 6) ** 2
    band_top = C - R + int(BAND * 2 * R)
    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)
    out = {}
    for name, sel in (("top", None), ("bottom", None)):
        mask = inside.copy()
        if name == "top":
            mask[band_top:] = False
            mask[:C - R] = False
        else:
            mask[:SIZE - band_top] = False
        binv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 51, 8)
        binv[~mask] = 0
        n, _, stats, _ = cv2.connectedComponentsWithStats(binv, 8)
        cnt = sum(1 for i in range(1, n) if stats[i][4] >= 40 and stats[i][3] >= 8)
        out[name] = cnt
    return out


def main():
    src, out_base, out_final, station, brand = sys.argv[1:6]
    json_path = sys.argv[6] if len(sys.argv) > 6 else None
    t0 = time.time()
    rgb = np.array(Image.open(src).convert("RGB"))
    rep = {"station": station, "src": src, "timings": {}}

    # 门禁 1：圆形度（在原图上）
    t = time.time()
    circ = G.circularity(rgb)
    rep["circularity"] = circ
    rep["timings"]["circularity"] = round(time.time() - t, 2)
    rep["gate_circularity"] = bool(circ["ok"])

    # 归一化 + 色带压平 -> 底图
    t = time.time()
    rgba = normalize(rgb)
    rgba = flatten_bands(rgba)
    Image.fromarray(rgba).save(out_base)
    rep["timings"]["normalize"] = round(time.time() - t, 2)

    # 门禁 2：文字（底图场景区）
    t = time.time()
    rows, glyphs, cands, grid = G.detect_text(rgba)
    rep["text_gate"] = {"rows": rows, "glyphs": glyphs, "grid_flag": grid,
                        "candidates": len(cands),
                        "pass": rows <= GATE_TEXT_ROWS}
    rep["timings"]["text_gate"] = round(time.time() - t, 2)

    # 门禁 3：主体裁断
    t = time.time()
    tr = G.truncation(rgba)
    rep["truncation"] = {"count": tr["truncated_subjects"], "ok": tr["ok"],
                         "comps": tr["comps"]}
    rep["timings"]["truncation"] = round(time.time() - t, 2)

    sat0, satp = G.mean_saturation(rgba)
    rep["saturation_pct_raw"] = satp
    rep["band_runs_rel_y"] = G.band_runs(rgba)

    rep["gate_all_pass"] = bool(rep["gate_circularity"] and rep["text_gate"]["pass"]
                                and tr["ok"])

    # 嵌字 + 复检
    t = time.time()
    final = emboss(rgba.copy(), station, brand)
    Image.fromarray(final).save(out_final)
    rep["timings"]["emboss"] = round(time.time() - t, 2)
    bt = band_text_check(final)
    scene_rows, scene_glyphs, _, _ = G.detect_text(final)
    rep["recheck"] = {
        "band_text_top": bt["top"], "band_text_bottom": bt["bottom"],
        "scene_text_rows_after": scene_rows,
        "pass": bt["top"] >= 1 and bt["bottom"] >= 1 and scene_rows <= GATE_TEXT_ROWS,
    }
    rep["total_sec"] = round(time.time() - t0, 2)
    if json_path:
        json.dump(rep, open(json_path, "w"), ensure_ascii=False, indent=2)
    print(json.dumps(rep, ensure_ascii=False))


if __name__ == "__main__":
    main()
