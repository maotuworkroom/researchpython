#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""正/负对照集生成器（确定性，种子固定）。

正对照：在真实印章底图（无字、色带已压平）的场景区域嵌入假站名，5 种变体：
  vivid    鲜色（高饱和红/黄/蓝），不透明，字号 ~130
  dark     深色（近黑），不透明，字号 ~130
  semi     半透明（alpha=105）浅色，字号 ~130
  small    小字号（~48）深色
  rotated  旋转 ±12~18°，字号 ~110
6 底图 × 5 变体 = 30 例（>=20 要求）。

负对照（20 例）：
  neg_base_<code>   底图原件（6）
  neg_norm_<code>   色彩归一后（6）
  neg_final_<code>  只在色带嵌字（站名+JR）的最终图（6）——检验色带排除
  neg_sat_<code>    目标饱和度 45 的归一变体（2）

用法:
  本地(有底图): python3 make_positives.py --bases-dir ../stampgate --out-dir positive_cases \
      --stations ../stampgate/stations_cn.json
  云端(从原图): python3 make_positives.py --from-raw --raw-dir ../../bench/inputs \
      --out-dir positive_cases --stations ../../bench/stations_cn.json
"""
import argparse
import hashlib
import json
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ocr_gate as OG  # noqa: E402

SIZE, C, R, BAND = OG.SIZE, OG.C, OG.R, OG.BAND

FONT_CANDIDATES = [
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 0),
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 1),
    ("/System/Library/Fonts/Hiragino Sans GB.ttc", 0),
    ("/System/Library/Fonts/PingFang.ttc", 1),
]

# 6 个“胜出”底图对应的原图与站名（与 stampgate/results.json 一致）
BASE_RAWS = [
    ("1005", "raw_1005_r2.jpg", "稚内駅"),
    ("1031", "raw_1031.jpg", "青森駅"),
    ("1091", "raw_1091.jpg", "下灘駅"),
    ("1125", "raw_1125_r2.jpg", "美瑛駅"),
    ("1256", "raw_1256.jpg", "東京駅"),
    ("1443", "raw_1443.jpg", "大歩危駅"),
]

VARIANTS = ["vivid", "dark", "semi", "small", "rotated"]
VIVID_COLORS = [(228, 32, 32), (255, 196, 0), (24, 132, 231)]


def pick_font(size):
    for path, idx in FONT_CANDIDATES:
        if not os.path.exists(path):
            continue
        try:
            return ImageFont.truetype(path, size, index=idx), path
        except Exception:
            continue
    raise RuntimeError("no CJK font available")


def fit_font(text, max_w, max_h, start):
    size = start
    while size > 12:
        f, p = pick_font(size)
        l, t, r, b = f.getbbox(text)
        if (r - l) <= max_w and (b - t) <= max_h:
            return f, p, size
        size -= 4
    f, p = pick_font(12)
    return f, p, 12


def _bbox_fits(ins, cx, cy, w, h, margin=6):
    pts = [(cx - w / 2 + margin, cy - h / 2 + margin),
           (cx + w / 2 - margin, cy - h / 2 + margin),
           (cx - w / 2 + margin, cy + h / 2 - margin),
           (cx + w / 2 - margin, cy + h / 2 - margin),
           (cx, cy - h / 2 + margin), (cx, cy + h / 2 - margin),
           (cx - w / 2 + margin, cy), (cx + w / 2 - margin, cy)]
    for x, y in pts:
        xi, yi = int(x), int(y)
        if not (0 <= xi < SIZE and 0 <= yi < SIZE):
            return False
        if not ins[yi, xi]:
            return False
    return True


def place_center(rng, ins, w, h):
    band_top = C - R + int(BAND * 2 * R)
    band_bot = SIZE - band_top
    for _ in range(200):
        cy = rng.randint(band_top + h // 2 + 30, band_bot - h // 2 - 30)
        dy = abs(C - cy)
        chord = 2 * int(np.sqrt(max(R * R - dy * dy, 0)))
        half = max(int(chord * 0.32), w // 2 + 20)
        cx = rng.randint(C - half, C + half)
        if _bbox_fits(ins, cx, cy, w, h):
            return cx, cy
    raise RuntimeError("placement failed")


def render_text(text, font, color, alpha):
    l, t, r, b = font.getbbox(text)
    w, h = r - l, b - t
    pad = int(h * 0.35) + 8
    layer = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.text((pad - l, pad - t), text, font=font, fill=color + (alpha,))
    return layer, w, h


def embed(rgba, text, variant, rng, ins):
    """把假站名按变体嵌入场景区域，返回新 RGBA 与参数记录。"""
    out = rgba.copy()
    base_img = Image.fromarray(out)
    if variant == "vivid":
        color = VIVID_COLORS[rng.randrange(len(VIVID_COLORS))]
        f, fp, fs = fit_font(text, 900, 240, 130)
        alpha = 255
        angle = 0
    elif variant == "dark":
        color = (24, 27, 38)
        f, fp, fs = fit_font(text, 900, 240, 130)
        alpha = 255
        angle = 0
    elif variant == "semi":
        color = (250, 250, 250)
        f, fp, fs = fit_font(text, 900, 240, 130)
        alpha = 105
        angle = 0
    elif variant == "small":
        color = (24, 27, 38)
        f, fp, fs = fit_font(text, 420, 90, 48)
        alpha = 255
        angle = 0
    elif variant == "rotated":
        color = (200, 30, 30)
        f, fp, fs = fit_font(text, 800, 220, 110)
        alpha = 255
        angle = round(rng.uniform(12, 18) * rng.choice([-1, 1]), 1)
    else:
        raise ValueError(variant)

    layer, w, h = render_text(text, f, color, alpha)
    if angle:
        layer = layer.rotate(angle, expand=True, resample=Image.BICUBIC)
    lw, lh = layer.size
    cx, cy = place_center(rng, ins, lw, lh)
    base_img.alpha_composite(layer, (int(cx - lw / 2), int(cy - lh / 2)))
    out = np.array(base_img)
    return out, {"variant": variant, "name": text, "font": os.path.basename(fp),
                 "font_size": fs, "color": list(color), "alpha": alpha,
                 "angle": angle, "center": [int(cx), int(cy)],
                 "layer_wh": [int(lw), int(lh)]}


def base_from_raw(raw_path):
    """原图 -> 归一化 + 色带压平 底图（与 pipeline 同口径）。"""
    import cv2
    try:
        import pipeline as P
    except ImportError:
        sys.path.insert(0, os.path.join(HERE))
        import pipeline as P
    rgb = np.array(Image.open(raw_path).convert("RGB"))
    rgba = P.normalize(rgb)
    rgba = P.flatten_bands(rgba)
    return rgba


def sha1_10(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:10]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases-dir", help="已有底图目录（含 base_<code>.png）")
    ap.add_argument("--from-raw", action="store_true", help="从原图现算底图")
    ap.add_argument("--raw-dir")
    ap.add_argument("--stations", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--skip-negatives", action="store_true")
    args = ap.parse_args()

    stations = json.load(open(args.stations))
    names = [stations[i]["name"] for i in range(5, 5 + 5 * 30, 5)]  # 30 个假站名
    os.makedirs(args.out_dir, exist_ok=True)
    ins = OG.scene_mask()
    manifest = {"seed_rule": "random.Random(f'{code}_{variant}')",
                "variants": VARIANTS, "positives": [], "negatives": []}
    bases = {}

    for k, (code, rawf, st_name) in enumerate(BASE_RAWS):
        if args.from_raw:
            rgba = base_from_raw(os.path.join(args.raw_dir, rawf))
        else:
            rgba = np.array(Image.open(
                os.path.join(args.bases_dir, f"base_{code}.png")).convert("RGBA"))
        bases[code] = (rgba, st_name)

        for variant in VARIANTS:
            rng = random.Random(f"{code}_{variant}")
            fake = names[k * len(VARIANTS) + VARIANTS.index(variant)]
            out, info = embed(rgba, fake, variant, rng, ins)
            fn = f"pos_{code}_{variant}.png"
            Image.fromarray(out).save(os.path.join(args.out_dir, fn))
            manifest["positives"].append(
                {"file": fn, "base": code, "truth": "text", **info})
            print(f"[pos] {fn} name={fake} fs={info['font_size']} "
                  f"a={info['alpha']} ang={info['angle']}")

        if not args.skip_negatives:
            fn = f"neg_base_{code}.png"
            Image.fromarray(rgba).save(os.path.join(args.out_dir, fn))
            manifest["negatives"].append({"file": fn, "base": code,
                                          "truth": "clean", "kind": "base"})

    if not args.skip_negatives:
        import color_norm as CN
        for code, (rgba, st_name) in bases.items():
            fn = f"neg_norm_{code}.png"
            Image.fromarray(CN.normalize_color(rgba)).save(
                os.path.join(args.out_dir, fn))
            manifest["negatives"].append({"file": fn, "base": code,
                                          "truth": "clean", "kind": "norm"})
            fn = f"neg_final_{code}.png"
            try:
                import pipeline as P
                final = P.emboss(rgba.copy(), st_name, "JR")
            except Exception:
                sys.path.insert(0, HERE)
                import pipeline as P
                final = P.emboss(rgba.copy(), st_name, "JR")
            Image.fromarray(final).save(os.path.join(args.out_dir, fn))
            manifest["negatives"].append({"file": fn, "base": code,
                                          "truth": "clean", "kind": "final_band_text"})
        for code in ("1005", "1256"):
            rgba, _ = bases[code]
            fn = f"neg_sat_{code}.png"
            Image.fromarray(CN.normalize_color(rgba, target_sat_pct=45.0)).save(
                os.path.join(args.out_dir, fn))
            manifest["negatives"].append({"file": fn, "base": code,
                                          "truth": "clean", "kind": "norm_sat45"})

    for entry in manifest["positives"] + manifest["negatives"]:
        entry["sha1"] = sha1_10(os.path.join(args.out_dir, entry["file"]))
    json.dump(manifest, open(os.path.join(args.out_dir, "manifest.json"), "w"),
              ensure_ascii=False, indent=1)
    print(f"positives={len(manifest['positives'])} "
          f"negatives={len(manifest['negatives'])} -> {args.out_dir}")


if __name__ == "__main__":
    main()
