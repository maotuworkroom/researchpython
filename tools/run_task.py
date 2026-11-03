#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重活执行器：批量处理 bench/inputs/ 下的图像，产出 out/summary.json + 逐 job 报告。

任务:
  stamp-pipeline  印章全流程（圆形度门禁->归一化->文字门禁->裁断->嵌字->复检），纯 CPU
  color-norm      色彩归一批量（饱和度/色相归一），纯 CPU
  stamp-ocr-gate  tesseract OCR 文字门禁：调参(tune) / 30 站回归(regress)，需 runner 装 tesseract

参数 JSON:
  repeat   每张图重复次数（放大批量用）, 默认 1
  workers  进程池大小, 默认 1
  images   指定输入文件名列表, 默认取 bench/inputs 全部
  target_sat  color-norm 目标饱和度百分比, 默认 60
  mode     stamp-ocr-gate: "tune"（预处理网格调参）或 "regress"（30 站回归）
  configs  tune 用：门禁配置列表（缺省为裁切+缩放/二值化网格, psm=11/lang=jpn+eng/conf>=60）
  limit    tune 用：>0 时每类对照只取前 N 例（快速探测）
  gate     regress 用：选定的门禁配置, 缺省 {"scale":0.5,"binarize":"otsu"}
  n_stations  regress 站数, 默认 30

产物: out/summary.json（机器信息+吞吐+逐 job 报告）、out/sample_*.png（1 张视觉验证图）
      stamp-ocr-gate 另出 out/tune_report.json / out/regress_report.json
"""
import argparse
import contextlib
import io
import json
import multiprocessing as mp
import os
import platform
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import gate_decision as GD  # noqa: E402  阈值配置 + 失败原因分类

SAMPLE_TAG = None
_STATIONS = None


def _patch_fonts():
    """Linux runner 上用 Noto CJK；macOS 保留原候选。"""
    import pipeline as P
    P.FONT_CANDIDATES = [
        ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
        ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 0),
    ] + list(P.FONT_CANDIDATES)


def _init_worker(sample_tag=None):
    global SAMPLE_TAG
    SAMPLE_TAG = sample_tag
    _patch_fonts()


def _station_name(code):
    global _STATIONS
    if _STATIONS is None:
        _STATIONS = {}
        p = os.path.join(ROOT, "bench", "stations_cn.json")
        if os.path.exists(p):
            try:
                d = json.load(open(p))
                if isinstance(d, list):
                    _STATIONS = {str(x.get("id")): x.get("name") for x in d}
                elif isinstance(d, dict):
                    _STATIONS = {str(k): (v.get("name") if isinstance(v, dict) else v)
                                   for k, v in d.items()}
            except Exception:
                pass
    return _STATIONS.get(code) or f"駅{code}"


def _cleanup(paths, tag, keep_sample, sample_dst):
    if tag == SAMPLE_TAG:
        for src_p, name in ((paths[0], "sample_base.png"), (paths[1], "sample_final.png")):
            if os.path.exists(src_p):
                shutil.move(src_p, os.path.join(ROOT, "out", name))
    else:
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass


def _job_pipeline(job):
    src, tag = job
    import pipeline as P
    code = os.path.basename(src).split("_")[1].split(".")[0]
    scratch = os.path.join(ROOT, "out", "_scratch")
    base = os.path.join(scratch, f"base_{tag}.png")
    final = os.path.join(scratch, f"final_{tag}.png")
    rep_path = os.path.join(scratch, f"rep_{tag}.json")
    argv_save = sys.argv
    sys.argv = ["pipeline.py", src, base, final, _station_name(code), "JR", rep_path]
    t0 = time.time()
    err = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            P.main()
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    finally:
        sys.argv = argv_save
    rep = None
    if os.path.exists(rep_path):
        full = json.load(open(rep_path))
        rep = {k: full.get(k) for k in ("circularity", "text_gate", "truncation",
                                        "gate_all_pass", "recheck", "saturation_pct_raw",
                                        "total_sec", "timings")}
        try:
            os.remove(rep_path)
        except OSError:
            pass
    _cleanup((base, final), tag, True, None)
    return {"tag": tag, "src": os.path.basename(src), "sec": round(time.time() - t0, 2),
            "error": err, "report": rep}


def _job_colornorm(job):
    src, tag = job
    import numpy as np
    from PIL import Image
    import color_norm as CN
    import gate_lib as G
    scratch = os.path.join(ROOT, "out", "_scratch")
    dst = os.path.join(scratch, f"norm_{tag}.png")
    t0 = time.time()
    err, info = None, {}
    try:
        rgba = np.array(Image.open(src).convert("RGBA"))
        s_before, _ = G.mean_saturation(rgba)
        out = CN.normalize_color(rgba)
        s_after, _ = G.mean_saturation(out)
        Image.fromarray(out).save(dst)
        info = {"sat_before": s_before, "sat_after": s_after}
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    _cleanup((dst, dst), tag, True, None)
    return {"tag": tag, "src": os.path.basename(src), "sec": round(time.time() - t0, 2),
            "error": err, "report": info}


# ---------------- stamp-ocr-gate（tesseract 文字门禁） ----------------

# tune 缺省网格：psm=11 / lang=jpn+eng / conf>=60 固定，变预处理（裁切+缩放/二值化）
# 注：全幅 2048 灰度直跑在本 runner 上病态慢（>140s/图），故默认先裁场景 bbox
# 并降采样；"放大 2x"（scale=2）作为可选档保留。
DEFAULT_TUNE_GRID = [
    {"name": "c05gray", "scale": 0.5},
    {"name": "c05otsu", "scale": 0.5, "binarize": "otsu"},
    {"name": "c10otsu", "scale": 1.0, "binarize": "otsu"},
    {"name": "c05adp", "scale": 0.5, "binarize": "adaptive"},
]

NOTO_FONT_PATHS = [
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 0),
]


def _job_ocr_eval(job):
    """单图 OCR 门禁评估（对照集 tune 用）：走 gate_decision 出 verdict+reason。"""
    path, truth, label, gate_cfg, expected_text, expected_bbox = job
    import numpy as np
    from PIL import Image
    import gate_decision as GD
    import ocr_gate as OG
    t0 = time.time()
    cfg = GD.load_config()
    err, res = None, None
    try:
        rgba = np.array(Image.open(path).convert("RGBA"))
        cfg2 = dict(cfg)
        og = dict(GD.ocr_cfg_from(cfg))
        og.update({k: v for k, v in gate_cfg.items() if v is not None})
        ocr = OG.ocr_scene_text(rgba, **og)
        verdict, reason, metrics = GD.classify(ocr, cfg2, expected_text,
                                               expected_bbox, rgba)
        res = {"verdict": verdict, "reason": reason, "metrics": metrics,
               "ocr_sec": ocr["ocr_sec"], "hits": ocr["hits"][:6]}
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return {"label": label, "truth": truth, "sec": round(time.time() - t0, 2),
            "error": err,
            "verdict": res["verdict"] if res else None,
            "reason": res["reason"] if res else None,
            "metrics": res["metrics"] if res else None,
            "hits": res["hits"] if res else [],
            "ocr_sec": res["ocr_sec"] if res else None,
            "config_version": cfg["config_version"], "config_sha256": cfg["_sha256"]}


def _font_missing(text):
    """字体缺字门禁（fontTools cmap，与 font_scan.py 同口径）。"""
    try:
        from fontTools.ttLib import TTCollection
    except ImportError:
        return None, "fontTools 不可用"
    for path, idx in NOTO_FONT_PATHS:
        if os.path.exists(path):
            ttc = TTCollection(path, lazy=True)
            cmap = ttc.fonts[idx].getBestCmap()
            return [ch for ch in sorted(set(text)) if ord(ch) not in cmap], path
    return None, "无 CJK 字体文件"


def _build_cases(scratch):
    """云端重建正/负对照集（与本机同脚本同种子）。返回 (cases_dir, manifest)。"""
    import make_positives as MP
    cases_dir = os.path.join(scratch, "cases")
    argv_save = sys.argv
    sys.argv = ["make_positives.py", "--from-raw",
                "--raw-dir", os.path.join(ROOT, "bench", "inputs"),
                "--stations", os.path.join(ROOT, "bench", "stations_cn.json"),
                "--out-dir", cases_dir]
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            MP.main()
    finally:
        sys.argv = argv_save
    return cases_dir, json.load(open(os.path.join(cases_dir, "manifest.json")))


def run_ocr_tune(params, out_dir, scratch):
    """调参：对全部对照集跑一组门禁配置，输出召回/误报曲线数据。"""
    import pytesseract
    workers = max(1, int(params.get("workers", 4)))
    configs = params.get("configs") or DEFAULT_TUNE_GRID
    t_cases0 = time.time()
    cases_dir, mani = _build_cases(scratch)
    cases_sec = time.time() - t_cases0
    try:
        tess_ver = str(pytesseract.get_tesseract_version())
        langs = pytesseract.get_languages(config="")
    except Exception as e:
        tess_ver, langs = f"error: {e}", []
    report = {"tesseract": tess_ver, "langs": langs,
              "cases": {"positives": len(mani["positives"]),
                        "negatives": len(mani["negatives"]),
                        "build_sec": round(cases_sec, 1)},
              "fixed": {"psm": 11, "lang": "jpn+eng", "conf_thr": 60},
              "configs": []}
    limit = int(params.get("limit", 0))  # >0: 每类取前 N 例（快速探测用）
    pos_only = bool(params.get("positives_only", False))  # 只跑正例（提速探测召回）
    if limit > 0:
        pos_sel = mani["positives"][::max(1, len(mani["positives"]) // limit)][:limit]
        neg_sel = mani["negatives"][::max(1, len(mani["negatives"]) // limit)][:limit]
        report["limit"] = {"positives": len(pos_sel), "negatives": len(neg_sel)}
    else:
        pos_sel, neg_sel = mani["positives"], mani["negatives"]
    if pos_only:
        neg_sel = []
        report["positives_only"] = True

    def _bbox(e):
        if "center" not in e or "layer_wh" not in e:
            return None
        cx, cy = e["center"]
        w, h = e["layer_wh"]
        return (cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2)

    t0 = time.time()
    for cfg in configs:
        gate_cfg = {k: v for k, v in cfg.items() if k != "name"}
        jobs = [(os.path.join(cases_dir, e["file"]), e["truth"], e["file"], gate_cfg,
                 e.get("name", ""), _bbox(e)) for e in pos_sel + neg_sel]
        if workers > 1:
            with mp.Pool(workers) as pool:
                results = pool.map(_job_ocr_eval, jobs)
        else:
            results = [_job_ocr_eval(j) for j in jobs]
        pos = [r for r in results if r["truth"] == "text"]
        neg = [r for r in results if r["truth"] == "clean"]
        # 召回 = 正例被判 fail；误报 = 负例被判 fail
        rec = sum(1 for r in pos if r["verdict"] == "fail")
        fp = [r for r in neg if r["verdict"] == "fail"]
        reasons = {}
        for r in results:
            reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        report["configs"].append({
            "name": cfg.get("name", json.dumps(gate_cfg, ensure_ascii=False)),
            "config": gate_cfg,
            "recall": f"{rec}/{len(pos)}", "recall_pct": round(100 * rec / max(len(pos), 1), 1),
            "fp": f"{len(fp)}/{len(neg)}",
            "reason_hist": reasons,
            "missed": [{"label": r["label"], "reason": r["reason"],
                        "n_hits": (r["metrics"] or {}).get("n_hits")} for r in pos
                       if r["verdict"] != "fail"],
            "false_pos": [{"label": r["label"], "reason": r["reason"],
                           "hits": [h["text"] for h in r["hits"]]} for r in fp],
            "ocr_sec_sum": round(sum(r["ocr_sec"] or 0 for r in results), 1),
            "ocr_sec_max": round(max([r["ocr_sec"] or 0 for r in results] or [0]), 1),
            "errors": [r["error"] for r in results if r["error"]][:5],
        })
        report["wall_sec"] = round(time.time() - t0, 1)
        with open(os.path.join(out_dir, "tune_report.json"), "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=1)  # 逐组落盘防超时丢失
    # 样图：1 正 1 负，供人工抽查
    for src, dst in ((os.path.join(cases_dir, mani["positives"][0]["file"]),
                      os.path.join(out_dir, "sample_positive.png")),
                     (os.path.join(cases_dir, "neg_final_1005.png"),
                      os.path.join(out_dir, "sample_negative.png"))):
        if os.path.exists(src):
            shutil.copy(src, dst)
    return report


def _job_ocr_regress(job):
    """30 站回归单 job：全流程 + 新旧文字门禁对照。"""
    src, station_id, station_name, gate_cfg, keep_sample = job
    import numpy as np
    from PIL import Image
    import pipeline as P
    import gate_lib as G
    import ocr_gate as OG
    import gate_decision as GD
    t0 = time.time()
    rep = {"station_id": station_id, "station": station_name,
           "src": os.path.basename(src), "timings": {}}
    err = None
    try:
        rgb = np.array(Image.open(src).convert("RGB"))
        t = time.time()
        circ = G.circularity(rgb)
        rep["timings"]["circularity"] = round(time.time() - t, 2)
        rep["gate_circularity"] = bool(circ["ok"])
        rep["circ_score"] = circ["score"]

        t = time.time()
        rgba = P.normalize(rgb)
        rgba = P.flatten_bands(rgba)
        rep["timings"]["normalize"] = round(time.time() - t, 2)

        t = time.time()
        rows = G.detect_text(rgba)[0]
        rep["timings"]["text_gate_old"] = round(time.time() - t, 2)
        rep["old_text_rows"] = rows
        rep["old_text_pass"] = bool(rows <= 0)

        t = time.time()
        ocr_base = OG.ocr_scene_text(rgba, **gate_cfg)
        rep["timings"]["text_gate_ocr"] = round(time.time() - t, 2)
        vb, rb, mb = GD.classify(ocr_base, GD.load_config(), "", None, rgba)
        rep["ocr_base"] = {"verdict": vb, "reason": rb, "n_hits": mb["n_hits"],
                           "hits": [h["text"] for h in ocr_base["hits"][:6]],
                           "prep_sha1": ocr_base["prep_sha1"],
                           "ocr_sec": ocr_base["ocr_sec"]}

        t = time.time()
        tr = G.truncation(rgba)
        rep["timings"]["truncation"] = round(time.time() - t, 2)
        rep["gate_truncation"] = bool(tr["ok"])
        rep["trunc_count"] = tr["truncated_subjects"]

        _, satp = G.mean_saturation(rgba)
        rep["sat_pct"] = satp

        missing, font_path = _font_missing(station_name + "JR")
        rep["font_missing"] = missing
        rep["gate_font"] = bool(missing is not None and len(missing) == 0)

        t = time.time()
        final = P.emboss(rgba.copy(), station_name, "JR")
        rep["timings"]["emboss"] = round(time.time() - t, 2)
        bt = P.band_text_check(final)
        ocr_final = OG.ocr_scene_text(final, **gate_cfg)
        vf, rf, mf = GD.classify(ocr_final, GD.load_config(), "", None, final)
        rep["recheck"] = {"band_top": bt["top"], "band_bottom": bt["bottom"],
                          "ocr_final_verdict": vf, "ocr_final_reason": rf,
                          "ocr_final_hits": mf["n_hits"],
                          "prep_sha1": ocr_final["prep_sha1"],
                          "pass": bool(bt["top"] >= 1 and bt["bottom"] >= 1
                                       and vf == "pass")}
        rep["gate_all_pass"] = bool(rep["gate_circularity"] and vb == "pass"
                                    and rep["gate_truncation"] and rep["gate_font"])
        rep["fail_reasons"] = [x for x in [
            None if rep["gate_circularity"] else "circularity",
            rb if vb != "pass" else None,
            None if rep["gate_truncation"] else "truncation",
            None if rep["gate_font"] else "font_missing_glyph",
            None if rep["recheck"]["pass"] else ("band_recheck:" + str(rf)),
        ] if x]
        if keep_sample:
            rep["_final"] = final  # 仅首 job 留样，避免大数组过 pickle
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    rep["sec"] = round(time.time() - t0, 2)
    rep["error"] = err
    return rep


def run_ocr_regress(params, out_dir, scratch):
    """8 图 × repeat 回归：四道门禁（新 OCR 文字门禁）+ 新旧文字门禁对照。"""
    workers = max(1, int(params.get("workers", 4)))
    repeat = max(1, int(params.get("repeat", 3)))
    gate_cfg = params.get("gate") or {}
    stations = json.load(open(os.path.join(ROOT, "bench", "stations_cn.json")))
    inputs_dir = os.path.join(ROOT, "bench", "inputs")
    raws = sorted(f for f in os.listdir(inputs_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))[:8]
    jobs, keys = [], []
    for k in range(repeat):
        for i, raw in enumerate(raws):
            st = stations[(i * repeat + k) % len(stations)]
            keys.append(f"{raw}#r{k}")
            jobs.append((os.path.join(inputs_dir, raw), st["id"], st["name"],
                         gate_cfg, (k == 0 and i == 0)))

    t0 = time.time()
    if workers > 1:
        with mp.Pool(workers) as pool:
            results = pool.map(_job_ocr_regress, jobs)
    else:
        results = [_job_ocr_regress(j) for j in jobs]
    wall = time.time() - t0
    for key, r in zip(keys, results):
        r["key"] = key

    sample_kept = False
    for r in results:
        fin = r.pop("_final", None)
        if fin is not None and not sample_kept:
            from PIL import Image
            Image.fromarray(fin).save(os.path.join(out_dir, "sample_regress_final.png"))
            sample_kept = True
    errs = [r for r in results if r["error"]]
    fp_rate = sum(1 for r in results if r["gate_all_pass"]) / max(len(results), 1)
    gdcfg = GD.load_config()
    report = {
        "ok": len(errs) == 0 and sum(1 for r in results if r["gate_all_pass"]) == len(results),
        "gate_config": gate_cfg,
        "n_images": len(raws),
        "repeat": repeat,
        "n_jobs": len(results),
        "raws": raws,
        "config_version": gdcfg["config_version"],
        "config_sha256": gdcfg["_sha256"],
        "thresholds": gdcfg["decision"],
        "wall_sec": round(wall, 2),
        "throughput_job_per_sec": round(len(results) / wall, 3) if wall > 0 else 0,
        "first_pass_rate": round(fp_rate, 4),
        "first_pass_count": sum(1 for r in results if r["gate_all_pass"]),
        "gates": {
            "circularity_fail": [r["key"] for r in results if not r.get("gate_circularity")],
            "ocr_text_fail": [{"key": r["key"], "reason": r["ocr_base"].get("reason")}
                              for r in results
                              if r.get("ocr_base", {}).get("verdict") == "fail"],
            "old_text_fail": [r["key"] for r in results
                              if r.get("old_text_pass") is False],
            "truncation_fail": [r["key"] for r in results if not r.get("gate_truncation")],
            "font_fail": [r["key"] for r in results if not r.get("gate_font")],
            "recheck_fail": [{"key": r["key"], "reason": r["recheck"].get("ocr_final_reason")}
                             for r in results if not r["recheck"]["pass"]],
        },
        "avg_job_sec": round(sum(r["sec"] for r in results) / max(len(results), 1), 2),
        "jobs_failed": len(errs),
        "errors": [{"key": r["key"], "error": r["error"]} for r in errs],
        "jobs": results,
    }
    with open(os.path.join(out_dir, "regress_report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    return report


def machine_info():
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "cpu_count": mp.cpu_count(),
            "arch": (platform.processor() or getattr(os.uname(), "machine", "?"))}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    info["mem_mb"] = int(line.split()[1]) // 1024
                    break
    except Exception:
        pass
    try:
        import cv2
        import numpy
        info["cv2"], info["numpy"] = cv2.__version__, numpy.__version__
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True,
                    choices=["stamp-pipeline", "color-norm", "stamp-ocr-gate"])
    ap.add_argument("--params", default="{}")
    args = ap.parse_args()
    params = json.loads(args.params or "{}")
    inputs_dir = os.path.join(ROOT, "bench", "inputs")
    images = params.get("images") or sorted(
        f for f in os.listdir(inputs_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    repeat = max(1, int(params.get("repeat", 1)))
    workers = max(1, int(params.get("workers", 1)))
    out_dir = os.path.join(ROOT, "out")
    scratch = os.path.join(out_dir, "_scratch")
    os.makedirs(scratch, exist_ok=True)

    # ---- stamp-ocr-gate：独立流程（调参 / 回归） ----
    if args.task == "stamp-ocr-gate":
        _patch_fonts()
        mode = params.get("mode", "tune")
        t0 = time.time()
        if mode == "tune":
            report = run_ocr_tune(params, out_dir, scratch)
            head = {k: v for k, v in report.items() if k != "configs"}
            head["per_config"] = [{kk: c[kk] for kk in ("name", "recall", "fp",
                                                         "recall_pct", "ocr_sec_sum")}
                                  for c in report["configs"]]
        elif mode == "regress":
            report = run_ocr_regress(params, out_dir, scratch)
            head = {k: v for k, v in report.items() if k != "jobs"}
        else:
            sys.exit(f"未知 mode: {mode}（tune/regress）")
        summary = {"task": args.task, "mode": mode, "params": params,
                   "machine": machine_info(), "wall_sec": round(time.time() - t0, 2)}
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        shutil.rmtree(scratch, ignore_errors=True)
        print(json.dumps(head, ensure_ascii=False))
        # 不通过不静默：门禁失败或 job 异常 => 非零退出（产物已落盘，artifact if: always()）
        if mode == "regress" and not report.get("ok"):
            sys.exit("REGRESS_FAILED first_pass=%s/%s errors=%d reasons=%s" % (
                report["first_pass_count"], report["n_jobs"], report["jobs_failed"],
                {k: v for k, v in report["gates"].items() if v}))
        return

    jobs = [(os.path.join(inputs_dir, img), f"{os.path.splitext(img)[0]}_r{k}")
            for img in images for k in range(repeat)]
    sample_tag = jobs[0][1]
    jobfn = _job_pipeline if args.task == "stamp-pipeline" else _job_colornorm

    t0 = time.time()
    if workers > 1:
        with mp.Pool(workers, initializer=_init_worker, initargs=(sample_tag,)) as pool:
            results = pool.map(jobfn, jobs)
    else:
        _init_worker(sample_tag)
        results = [jobfn(j) for j in jobs]
    wall = time.time() - t0

    errs = [r for r in results if r["error"]]
    summary = {
        "task": args.task,
        "params": params,
        "machine": machine_info(),
        "jobs_total": len(jobs),
        "jobs_failed": len(errs),
        "wall_sec": round(wall, 2),
        "avg_job_sec": round(sum(r["sec"] for r in results) / max(len(results), 1), 3),
        "throughput_img_per_sec": round(len(jobs) / wall, 3) if wall > 0 else 0,
        "jobs": results,
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    shutil.rmtree(scratch, ignore_errors=True)
    print(json.dumps({k: v for k, v in summary.items() if k != "jobs"}, ensure_ascii=False))
    if errs:
        print("FAILED JOBS:", json.dumps(
            [{"tag": e["tag"], "error": e["error"]} for e in errs], ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
