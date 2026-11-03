#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""缺陷注入集：把 5 类嵌字缺陷逐个注入色带，验证门禁能给出可定位的失败原因。

缺陷（每类对应一个判定原因，命名与 gate_decision.REASONS 一致）：
  missing_glyph          少嵌一个字（站名截断 1 字）
  glyph_substitution     末字换成同长度的形近字（错字）
  position_out_of_bounds 站名整体横移，~35% 字宽越出圆盘
  low_contrast           字色贴近色带底色（灰度差 ~22，OCR 读不出）
  mask_eats_glyph        站名整体上移，~40% 字高跨出色带上边界（被门禁掩膜裁掉）

底座：真实原图 -> normalize -> flatten_bands；基准（无缺陷）= pipeline.emboss。
每类缺陷 6 张（6 个底图），共 30 张 + 6 张基准。OCR/判定在云端跑。

用法:
  python3 make_defects.py --from-raw --raw-dir ../repo/bench/inputs \
      --stations ../repo/bench/stations_cn.json --out-dir /tmp/sg2_defects
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "repo", "tools"))  # 本机没有 pipeline.py 时回退（排在后面）

import ocr_gate as OG  # noqa: E402

SIZE, C, R, BAND = OG.SIZE, OG.C, OG.R, OG.BAND
BAND_H = int(BAND * 2 * R)
DEFECTS = ("missing_glyph", "glyph_substitution", "position_out_of_bounds",
           "low_contrast", "mask_eats_glyph")

# 形近字（末字替换；权威性不依赖"像不像"，只要 OCR 读到非期望字即可触发字形替代）
HOMOGLYPH = {"駅": "馬", "内": "內", "稚": "雅", "前": "剪", "京": "凉", "大": "太",
             "歩": "步", "危": "厄", "青": "清", "森": "林", "灘": "难", "瑛": "英",
             "東": "柬", "美": "義", "尻": "尼", "焼": "燒", "山": "出"}


def _band_mid(side_top):
    return (C - R) + BAND_H // 2 if side_top else SIZE - (C - R) - BAND_H // 2


def _fg_for(rgba, side_top):
    import pipeline as P
    col = P.band_color(rgba, side_top)
    lum = 0.299 * col[0] + 0.587 * col[1] + 0.114 * col[2]
    return (31, 36, 48) if lum > 140 else (255, 255, 255)


def _fit(rgba, text, side_top):
    import pipeline as P
    y_mid = _band_mid(side_top)
    chord = 2 * int(np.sqrt(max(R * R - (C - y_mid) ** 2, 0)))
    start = int(BAND_H * 0.40) if side_top else int(BAND_H * 0.66)
    f = P.fit_font(text, int(chord * 0.78), int(BAND_H * 0.62), start)
    l, t, r, b = f.getbbox(text)
    return f, chord, (r - l, b - t, l, t)


def draw_band(rgba, side_top, text, color=None, alpha=255, dx=0, dy=0):
    """手动在色带中央嵌字（可横移/纵向偏移/改色），返回 (rgba, info)。"""
    if not text:
        return rgba, {"w": 0, "h": 0, "dx": dx, "dy": dy}
    img = Image.fromarray(rgba)
    d = ImageDraw.Draw(img)
    f, chord, (w, h, l, t) = _fit(rgba, text, side_top)
    y_mid = _band_mid(side_top) + dy
    d.text((C + dx - w / 2 - l, y_mid - h / 2 - t), text, font=f,
           fill=tuple(color if color else _fg_for(rgba, side_top)) + (alpha,))
    return np.array(img), {"w": int(w), "h": int(h), "dx": int(dx), "dy": int(dy),
                           "chord": int(chord), "color": list(color) if color else None,
                           "alpha": alpha}


def build_variants(rgba, station, brand="JR"):
    """返回 {name: (rgba, info)}：基准 + 5 类缺陷。"""
    import pipeline as P
    out = {"good": (P.emboss(rgba.copy(), station, brand), {"defect": None})}
    base = rgba.copy()

    # 1) 缺字：少嵌最后一字
    out["missing_glyph"] = (draw_band(base.copy(), False, station[:-1])[0],
                            {"dropped": station[-1]})
    # 2) 字形替代：末字形近替换
    sub = station[:-1] + HOMOGLYPH.get(station[-1], "馬")
    out["glyph_substitution"] = (draw_band(base.copy(), False, sub)[0], {"shown": sub})
    # 3) 位置越界：横移到 ~35% 字宽越出圆盘
    f, chord, (w, h, l, t) = _fit(base, station, False)
    dx = int(chord // 2 - 0.15 * w)
    out["position_out_of_bounds"] = (draw_band(base.copy(), False, station, dx=dx)[0],
                                     {"dx": dx, "w": w})
    # 4) 对比度不足：字色与底色灰度差 ~22
    import pipeline as P2
    col = P2.band_color(base, False).astype(int)
    low = tuple(int(np.clip(c + 22, 0, 255)) if c < 120 else int(np.clip(c - 22, 0, 255))
                for c in col)
    out["low_contrast"] = (draw_band(base.copy(), False, station, color=low)[0],
                           {"color": list(low), "band": [int(c) for c in col]})
    # 5) 遮罩吃掉字形：上移，使 ~40% 字高跨出色带内侧边界（被门禁掩膜裁掉）
    inner = OG.band_bottom_y()  # 下色带内侧边界 y
    y_mid = _band_mid(False)
    dy = int(inner - y_mid + 0.1 * h)
    out["mask_eats_glyph"] = (draw_band(base.copy(), False, station, dy=dy)[0],
                              {"dy": dy, "h": h, "inner_y": int(inner)})
    return out


def sha1_10(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-raw", action="store_true")
    ap.add_argument("--raw-dir")
    ap.add_argument("--bases-dir")
    ap.add_argument("--stations", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-bases", type=int, default=6)
    args = ap.parse_args()

    import pipeline as P
    stations = json.load(open(args.stations))
    names = [s["name"] for s in stations[:args.n_bases]]
    raws = sorted(f for f in os.listdir(args.raw_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))[:args.n_bases]
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {"defects": list(DEFECTS), "seed": "无随机（几何量直接计算）", "cases": []}

    for i, (raw, name) in enumerate(zip(raws, names)):
        rgb = np.array(Image.open(os.path.join(args.raw_dir, raw)).convert("RGB"))
        rgba = P.flatten_bands(P.normalize(rgb))
        for dtype, (img, info) in build_variants(rgba, name).items():
            fn = f"{dtype}_{i:02d}.png"
            Image.fromarray(img).save(os.path.join(args.out_dir, fn))
            manifest["cases"].append({
                "file": fn, "defect": None if dtype == "good" else dtype,
                "expected_reason": None if dtype == "good" else dtype,
                "station": name, "src": raw, **info,
                "sha1": sha1_10(os.path.join(args.out_dir, fn))})
        print(f"[base] {raw} station={name}")
    json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"cases={len(manifest['cases'])} -> {args.out_dir}")


if __name__ == "__main__":
    main()
