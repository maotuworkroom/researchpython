#!/usr/bin/env python3
# 色彩归一方案 (b)：后处理色板量化/色相归一 —— 保持明度 V，只压色相与饱和度
# 用法: nice -n 19 python3 color_norm.py IN.png OUT.png [--target-sat 60] [--hue-anchors 12]
import sys
import numpy as np
import cv2
from PIL import Image

import gate_lib as G


def normalize_color(rgba, target_sat_pct=60.0, hue_anchors=12, hue_snap_deg=10.0):
    """
    饱和度：两次迭代比例缩放，使圆内平均饱和度命中目标（保持 V 不变）。
    色相：向最近的 anchor 吸附（|Δh|<=hue_snap_deg 内软吸附），压缩色相离散度。
    """
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3]
    ins = G.inside_mask(0) & (alpha > 127)
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    target = target_sat_pct / 100 * 255
    S = hsv[:, :, 1]
    for _ in range(3):  # 迭代消除截断偏差
        mean = S[ins].mean()
        if mean < 1:
            break
        k = target / mean
        S = np.clip(S * k, 0, 255)
        hsv[:, :, 1] = S
    # 色相吸附（只动有色像素：S>20）
    if hue_anchors:
        step = 180.0 / hue_anchors
        H = hsv[:, :, 0]
        colored = S > 20
        nearest = np.round(H / step) * step % 180
        dh = (nearest - H + 90) % 180 - 90
        snap = colored & (np.abs(dh) <= hue_snap_deg)
        H[snap] = (H[snap] + 0.6 * dh[snap]) % 180  # 软吸附 60%
        hsv[:, :, 0] = H
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    rgb2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    out = rgba.copy()
    out[:, :, :3] = rgb2
    return out


def main():
    src, dst = sys.argv[1], sys.argv[2]
    target = 60.0
    anchors = 12
    args = sys.argv[3:]
    if "--target-sat" in args:
        target = float(args[args.index("--target-sat") + 1])
    if "--hue-anchors" in args:
        anchors = int(args[args.index("--hue-anchors") + 1])
    rgba = np.array(Image.open(src).convert("RGBA"))
    s_before, p_before = G.mean_saturation(rgba)
    out = normalize_color(rgba, target, anchors)
    s_after, p_after = G.mean_saturation(out)
    Image.fromarray(out).save(dst)
    print(f"{src}\t{p_before:.2f}%\t->\t{p_after:.2f}%")


if __name__ == "__main__":
    main()
