#!/usr/bin/env python3
# JR 印章门禁库：文字检测（连通域+笔画统计）、圆形度、主体裁断、饱和度
# 纯本地 numpy/OpenCV，无 OCR 依赖
import numpy as np
import cv2

SIZE = 2048
C = SIZE // 2
R = 1004
BAND = 0.15


def inside_mask(r_shrink=0):
    ys, xs = np.mgrid[0:SIZE, 0:SIZE]
    return (xs - C) ** 2 + (ys - C) ** 2 <= (R - r_shrink) ** 2


def scene_mask():
    """圆内、去掉上下 15% 色带后的场景区域。"""
    ins = inside_mask(8)
    band_top = C - R + int(BAND * 2 * R)
    band_bot = SIZE - band_top
    ins[:band_top] = False
    ins[band_bot:] = False
    return ins


# ---------------- 骨架 / 笔画统计 ----------------

_ELEM = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))


def skeletonize(mask_u8):
    skel = np.zeros_like(mask_u8)
    img = mask_u8.copy()
    while True:
        eroded = cv2.erode(img, _ELEM)
        opened = cv2.dilate(eroded, _ELEM)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def endpoints_junctions(skel_u8):
    k = np.ones((3, 3), dtype=np.float32)
    k[1, 1] = 0
    nbrs = cv2.filter2D((skel_u8 > 0).astype(np.uint8), -1, k, borderType=cv2.BORDER_CONSTANT)
    on = skel_u8 > 0
    return int((on & (nbrs == 1)).sum()), int((on & (nbrs >= 3)).sum())


def stroke_width(comp_mask_u8, skel_u8):
    dist = cv2.distanceTransform(comp_mask_u8, cv2.DIST_L2, 3)
    pts = dist[skel_u8 > 0]
    if len(pts) == 0:
        return float(comp_mask_u8.shape[0])
    return float(2.0 * np.median(pts))


def projection_peaks(mask_u8, axis):
    """投影直方图的峰数（轴=1 按列投影）。"""
    prof = (mask_u8 > 0).sum(axis=0 if axis == 1 else 1).astype(np.float32)
    if prof.max() == 0:
        return 0
    prof = np.convolve(prof, np.ones(3, dtype=np.float32) / 3, mode="same")
    thr = 0.25 * prof.max()
    peaks = 0
    above = False
    for v in prof:
        if v > thr and not above:
            peaks += 1
            above = True
        elif v <= thr:
            above = False
    return peaks


def comp_features(binv, x, y, w, h, area):
    """单个连通域的笔画统计特征。"""
    crop = binv[y:y + h, x:x + w]
    comp = (crop > 0).astype(np.uint8) * 255
    fill = area / float(w * h)
    skel = skeletonize(comp)
    ends, juncs = endpoints_junctions(skel)
    sw = stroke_width(comp, skel)
    px = projection_peaks(comp, 1)
    py = projection_peaks(comp, 0)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hull_area = 0.0
    holes = 0
    if cnts:
        hull_area = float(cv2.contourArea(cv2.convexHull(max(cnts, key=cv2.contourArea))))
        all_cnts, hier = cv2.findContours(comp, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is not None:
            holes = int((hier[0][:, 3] >= 0).sum())
    solidity = area / hull_area if hull_area > 0 else 1.0
    return {
        "x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": int(area),
        "fill": round(fill, 3),
        "aspect": round(w / h, 3),
        "stroke_ratio": round(sw / h, 3),
        "endpoints": ends, "junctions": juncs,
        "peaks_x": px, "peaks_y": py,
        "solidity": round(solidity, 3),
        "holes": holes,
    }


def glyph_score(f):
    """字形似然 0..1：方宽比、填充率、笔画细度、笔画结构、投影多峰、密度。"""
    s = 0.0
    a = f["aspect"]
    s += 1.0 if 0.45 <= a <= 2.2 else (0.4 if 0.3 <= a <= 3.0 else 0.0)
    s += 1.0 if 0.12 <= f["fill"] <= 0.68 else 0.0
    s += 1.0 if 0.045 <= f["stroke_ratio"] <= 0.38 else 0.0
    s += 1.0 if (f["endpoints"] + f["junctions"]) >= 2 else 0.3
    s += 1.0 if (f["peaks_x"] + f["peaks_y"]) >= 4 else (0.5 if f["peaks_x"] + f["peaks_y"] >= 3 else 0.2)
    s += 0.5 if 0.25 <= f["solidity"] <= 0.85 else 0.0
    dens = (f["endpoints"] + f["junctions"]) / max(f["h"], 1)
    s += 1.0 if dens <= 3.2 else 0.0
    return s / 6.5


# ---------------- 行聚簇 / 网格排除 ----------------

def cluster_rows(cands):
    """按 y 中心 + 高度相似把候选聚成行。"""
    cands = sorted(cands, key=lambda c: (c["y"] + c["h"] / 2, c["x"]))
    rows = []
    for c in cands:
        cy = c["y"] + c["h"] / 2
        placed = False
        for row in rows:
            med_h = np.median([k["h"] for k in row])
            med_cy = np.median([k["y"] + k["h"] / 2 for k in row])
            if abs(cy - med_cy) <= 0.55 * med_h and 0.55 <= c["h"] / med_h <= 1.8:
                row.append(c)
                placed = True
                break
        if not placed:
            rows.append([c])
    return [sorted(r, key=lambda c: c["x"]) for r in rows]


def is_grid(rows):
    """若 >=2 行尺寸一致且列位置对齐 => 窗格网格，判为非文字。"""
    sig = []
    for row in rows:
        if len(row) < 3:
            continue
        med_h = float(np.median([c["h"] for c in row]))
        med_w = float(np.median([c["w"] for c in row]))
        cols = tuple(round((c["x"] + c["w"] / 2) / max(med_w, 1)) for c in row)
        sig.append((med_h, med_w, cols, row))
    for i in range(len(sig)):
        for j in range(i + 1, len(sig)):
            h1, w1, c1, _ = sig[i]
            h2, w2, c2, _ = sig[j]
            if abs(h1 - h2) <= 0.12 * h1 and abs(w1 - w2) <= 0.12 * w1 and len(c1) == len(c2):
                align = np.mean([abs(a - b) for a, b in zip(c1, c2)])
                if align <= 0.6:
                    return True
    return False


def _sub_glyphlike(binv, x, y, w, h, area):
    f = comp_features(binv, x, y, w, h, area)
    dens = (f["endpoints"] + f["junctions"]) / max(h, 1)
    if f["stroke_ratio"] < 0.045 or f["aspect"] > 3.0 or dens > 3.2 \
            or (f["peaks_x"] + f["peaks_y"]) < 2:
        return None
    f["score"] = round(glyph_score(f), 3)
    return f if f["score"] >= 0.55 else None


def detect_text(rgba, min_score=0.55):
    """两级检测器：
    1) 闭运算把笔画碎片/相邻字合并为"文字行块"候选（尺寸/长宽比/填充过滤）；
    2) 行块内须含 >=2 个字形似然子连通域（笔画统计），且高度比 <=4、横向排布。
    返回 (疑似文字行数, 疑似字形数, 子域明细, 网格判定)。"""
    ins = scene_mask()
    gray = cv2.medianBlur(cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY), 5)
    rows_found = []
    subs_all = []
    for inv in (True, False):  # 深底浅字 + 浅底深字
        if inv:
            binv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 51, 8)
        else:
            binv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 51, 8)
        binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        binv[~ins] = 0
        linebin = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, np.ones((9, 25), np.uint8))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(linebin, 8)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if not (20 <= h <= 280 and w <= 900 and area >= 300):
                continue
            asp = w / max(h, 1)
            if not (1.2 <= asp <= 16):
                continue
            fill = area / float(w * h)
            if not (0.06 <= fill <= 0.65):
                continue
            crop = binv[y:y + h, x:x + w]
            ns, labs, sstats, _ = cv2.connectedComponentsWithStats(crop, 8)
            subs = []
            for j in range(1, ns):
                sx, sy, sw, sh, sarea = sstats[j]
                if sarea < 40 or sh < 12 or sw < 8:
                    continue
                g = _sub_glyphlike(crop, sx, sy, sw, sh, sarea)
                if g:
                    g["x"] += x
                    g["y"] += y
                    g["polarity"] = "dark" if inv else "light"
                    subs.append(g)
            if len(subs) < 2:
                continue
            strong = [s for s in subs if s["peaks_x"] + s["peaks_y"] >= 3]
            if len(strong) < 2:  # 需 >=2 个多笔画强证据子域
                continue
            if not any(s["peaks_y"] >= 2 for s in subs):  # 至少一个子域有横向多笔画
                continue
            hs = [s["h"] for s in subs]
            if max(hs) / max(min(hs), 1) > 4.0:
                continue
            if w < 1.5 * max(hs):
                continue
            rows_found.append({"bbox": [int(x), int(y), int(w), int(h)],
                               "n": len(subs), "pol": "dark" if inv else "light"})
            subs_all.extend(subs)
    # 跨极性去重（bbox 重叠的行块只计一次）
    rows_found.sort(key=lambda r: -(r["bbox"][2] * r["bbox"][3]))
    kept_rows = []
    for r in rows_found:
        dup = False
        for k in kept_rows:
            a, b = r["bbox"], k["bbox"]
            ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
            iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
            if ix * iy > 0.4 * min(a[2] * a[3], b[2] * b[3]):
                dup = True
                break
        if not dup:
            kept_rows.append(r)
    grid = False
    n_glyphs = sum(r["n"] for r in kept_rows)
    return len(kept_rows), n_glyphs, subs_all, grid


# ---------------- 圆形度 / 主体裁断 ----------------

def content_mask_rgb(rgb, sat_thr=25):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = (hsv[:, :, 1] > sat_thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    big = [c for c in cnts if cv2.contourArea(c) > 0.005 * m.shape[0] * m.shape[1]]
    if not big:
        return None, m
    return np.vstack(big), m


def circularity(rgb):
    """原图圆形度，双指标取小：
    a) 饱和内容掩膜拟合椭圆短/长轴（与试点口径一致）；
    b) 徽章外廓（非背景掩膜）bbox 短/长边比（抗低饱和天空干扰）。"""
    c, m = content_mask_rgb(rgb)
    if c is None or len(c) < 5:
        return {"fit_ellipse_aspect": 0.0, "roundness": 0.0, "ok": False}
    (ex, ey), (ea, eb), ang = cv2.fitEllipse(c)
    aspect = min(ea, eb) / max(ea, eb)
    rad = max(ea, eb) / 2
    roundness = cv2.countNonZero(m) / (np.pi * rad * rad) if rad > 0 else 0
    bg = rgb[8, 8].astype(int)
    diff = np.abs(rgb.astype(int) - bg).sum(axis=2)
    em = ((diff > 60).astype(np.uint8)) * 255
    em = cv2.morphologyEx(em, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    cnts, _ = cv2.findContours(em, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    emb = 1.0
    if cnts:
        x, y, w, h = cv2.boundingRect(max(cnts, key=cv2.contourArea))
        emb = min(w / h, h / w)
    score = min(float(aspect), float(emb))
    return {"fit_ellipse_aspect": round(float(aspect), 4),
            "emblem_bbox_ratio": round(float(emb), 4),
            "score": round(score, 4),
            "roundness": round(float(roundness), 4),
            "center": [round(float(ex), 1), round(float(ey), 1)],
            "ok": bool(score >= 0.97)}


def truncation(rgba, ring_pct=0.05, min_area_pct=0.015):
    """主体裁断：圆内显著紧凑连通域（地标/站房类，bbox 长宽比<=2.5）是否被圆边切断。
    排除：最大触边背景块；火车/海浪等横向装饰带（长宽比>2.5）。"""
    alpha = rgba[:, :, 3]
    rgb = rgba[:, :, :3]
    ins = inside_mask(0) & (alpha > 127)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat, val = hsv[:, :, 1], hsv[:, :, 2]
    content = ((sat > 30) | (val < 110) | ((sat <= 30) & (val >= 110) & (val <= 245))) & ins
    content = content.astype(np.uint8)
    content = cv2.morphologyEx(content, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(content, 8)
    circle_area = np.pi * R * R
    ring = inside_mask(0) & ~inside_mask(int(ring_pct * R))
    border = inside_mask(0) & ~inside_mask(3)  # 圆边 3px 内
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < min_area_pct * circle_area:
            continue
        comp = lab == i
        touches_border = bool((comp & border).any())
        in_ring = int((comp & ring).sum())
        comps.append({"idx": i, "area": int(area), "touches_border": touches_border,
                      "ring_px": in_ring, "bbox_aspect": round(w / max(h, 1), 2)})
    if not comps:
        return {"truncated_subjects": 0, "comps": [], "ok": True}
    comps.sort(key=lambda c: -c["area"])
    bg_idx = comps[0]["idx"] if comps[0]["touches_border"] else -1
    trunc = [c for c in comps
             if c["idx"] != bg_idx and c["touches_border"] and c["ring_px"] > 40
             and c["bbox_aspect"] <= 2.5]
    return {"truncated_subjects": len(trunc), "comps": comps[:8], "ok": len(trunc) == 0}


def mean_saturation(rgba):
    ins = inside_mask(0)
    hsv = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2HSV)
    s = float(hsv[:, :, 1][ins].mean())
    return round(s, 2), round(s / 255 * 100, 2)


def band_runs(rgba):
    ins = inside_mask(20)
    rgb = rgba[:, :, :3].astype(np.float32)
    y0, y1 = C - R, C + R
    std = []
    for y in range(y0, y1):
        px = rgb[y][ins[y]]
        std.append(float(px.std(axis=0).mean()) if len(px) > 50 else 999.0)
    solid = np.array(std) < 6.0
    runs, start = [], None
    for i, s_ in enumerate(solid):
        if s_ and start is None:
            start = i
        elif not s_ and start is not None:
            if i - start >= 10:
                runs.append((round(start / (2 * R), 4), round(i / (2 * R), 4)))
            start = None
    if start is not None and len(solid) - start >= 10:
        runs.append((round(start / (2 * R), 4), 1.0))
    return runs
