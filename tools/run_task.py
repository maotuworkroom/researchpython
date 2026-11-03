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
  configs  tune 用：门禁配置列表（缺省为 5 组预处理网格, psm=11/lang=jpn+eng/conf>=60）
  gate     regress 用：选定的门禁配置, 缺省 {"binarize":null,"upscale":1}
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

# tune 缺省网格：psm=11 / lang=jpn+eng / conf>=60 固定，只变预处理
DEFAULT_TUNE_GRID = [
    {"name": "gray"},
    {"name": "gray2x", "upscale": 2},
    {"name": "otsu", "binarize": "otsu"},
    {"name": "otsu2x", "binarize": "otsu", "upscale": 2},
    {"name": "adp", "binarize": "adaptive"},
]

NOTO_FONT_PATHS = [
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 0),
]


def _job_ocr_eval(job):
    """单图 OCR 门禁评估（对照集 tune 用）。"""
    path, truth, label, gate_cfg = job
    import numpy as np
    from PIL import Image
    import ocr_gate as OG
    t0 = time.time()
    err, out = None, None
    try:
        rgba = np.array(Image.open(path).convert("RGBA"))
        out = OG.ocr_scene_text(rgba, **gate_cfg)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return {"label": label, "truth": truth, "sec": round(time.time() - t0, 2),
            "error": err,
            "n_hits": out["n_hits"] if out else None,
            "pass": out["pass"] if out else None,
            "hits": out["hits"][:8] if out else [],
            "ocr_sec": out.get("ocr_sec") if out else None}


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
    t0 = time.time()
    for cfg in configs:
        gate_cfg = {k: v for k, v in cfg.items() if k != "name"}
        jobs = [(os.path.join(cases_dir, e["file"]), e["truth"], e["file"], gate_cfg)
                for e in mani["positives"] + mani["negatives"]]
        if workers > 1:
            with mp.Pool(workers) as pool:
                results = pool.map(_job_ocr_eval, jobs)
        else:
            results = [_job_ocr_eval(j) for j in jobs]
        pos = [r for r in results if r["truth"] == "text"]
        neg = [r for r in results if r["truth"] == "clean"]
        rec = sum(1 for r in pos if r["pass"] is False)
        fp = [r for r in neg if r["pass"] is False]
        report["configs"].append({
            "name": cfg.get("name", json.dumps(gate_cfg, ensure_ascii=False)),
            "config": gate_cfg,
            "recall": f"{rec}/{len(pos)}", "recall_pct": round(100 * rec / max(len(pos), 1), 1),
            "fp": f"{len(fp)}/{len(neg)}",
            "missed": [r["label"] for r in pos if r["pass"] is not False],
            "false_pos": [{"label": r["label"], "hits": r["hits"]} for r in fp],
            "ocr_sec_sum": round(sum(r["ocr_sec"] or 0 for r in results), 1),
            "errors": [r["error"] for r in results if r["error"]][:5],
        })
    report["wall_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(out_dir, "tune_report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
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
        rep["ocr_base"] = {"n_hits": ocr_base["n_hits"], "pass": ocr_base["pass"],
                           "hits": ocr_base["hits"][:6], "ocr_sec": ocr_base["ocr_sec"]}

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
        rep["recheck"] = {"band_top": bt["top"], "band_bottom": bt["bottom"],
                          "ocr_final_hits": ocr_final["n_hits"],
                          "pass": bool(bt["top"] >= 1 and bt["bottom"] >= 1
                                       and ocr_final["pass"])}
        rep["gate_all_pass"] = bool(rep["gate_circularity"] and ocr_base["pass"]
                                    and rep["gate_truncation"] and rep["gate_font"])
        if keep_sample:
            rep["_final"] = final  # 仅首 job 留样，避免大数组过 pickle
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    rep["sec"] = round(time.time() - t0, 2)
    rep["error"] = err
    return rep


def run_ocr_regress(params, out_dir, scratch):
    """n 站回归：8 张原图轮换 + 站名分布取样，四道门禁 + 新旧文字门禁对照。"""
    workers = max(1, int(params.get("workers", 4)))
    n = max(1, int(params.get("n_stations", 30)))
    gate_cfg = params.get("gate") or {"binarize": None, "upscale": 1}
    stations = json.load(open(os.path.join(ROOT, "bench", "stations_cn.json")))
    picks = [stations[i] for i in range(0, len(stations), len(stations) // n)][:n]
    inputs_dir = os.path.join(ROOT, "bench", "inputs")
    raws = sorted(f for f in os.listdir(inputs_dir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))
    jobs = [(os.path.join(inputs_dir, raws[i % len(raws)]),
             picks[i]["id"], picks[i]["name"], gate_cfg, i == 0) for i in range(n)]

    t0 = time.time()
    if workers > 1:
        with mp.Pool(workers) as pool:
            results = pool.map(_job_ocr_regress, jobs)
    else:
        results = [_job_ocr_regress(j) for j in jobs]
    wall = time.time() - t0

    sample_kept = False
    for r in results:
        fin = r.pop("_final", None)
        if fin is not None and not sample_kept:
            from PIL import Image
            Image.fromarray(fin).save(os.path.join(out_dir, "sample_regress_final.png"))
            sample_kept = True
    errs = [r for r in results if r["error"]]
    fp_rate = sum(1 for r in results if r["gate_all_pass"]) / max(len(results), 1)
    report = {
        "gate_config": gate_cfg,
        "n_stations": n,
        "raws_cycled": raws,
        "wall_sec": round(wall, 2),
        "throughput_station_per_sec": round(n / wall, 3) if wall > 0 else 0,
        "first_pass_rate": round(fp_rate, 4),
        "first_pass_count": sum(1 for r in results if r["gate_all_pass"]),
        "gates": {
            "circularity_fail": [r["station"] for r in results if not r.get("gate_circularity")],
            "ocr_text_fail": [r["station"] for r in results
                              if r.get("ocr_base", {}).get("pass") is False],
            "old_text_fail": [r["station"] for r in results
                              if r.get("old_text_pass") is False],
            "truncation_fail": [r["station"] for r in results
                                if not r.get("gate_truncation")],
            "font_fail": [r["station"] for r in results if not r.get("gate_font")],
        },
        "avg_job_sec": round(sum(r["sec"] for r in results) / max(len(results), 1), 2),
        "jobs_failed": len(errs),
        "stations": results,
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
            head = {k: v for k, v in report.items() if k != "stations"}
        else:
            sys.exit(f"未知 mode: {mode}（tune/regress）")
        summary = {"task": args.task, "mode": mode, "params": params,
                   "machine": machine_info(), "wall_sec": round(time.time() - t0, 2)}
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        shutil.rmtree(scratch, ignore_errors=True)
        print(json.dumps(head, ensure_ascii=False))
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
