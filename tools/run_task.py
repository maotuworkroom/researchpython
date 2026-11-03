#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重活执行器：批量处理 bench/inputs/ 下的图像，产出 out/summary.json + 逐 job 报告。

任务:
  stamp-pipeline  印章全流程（圆形度门禁->归一化->文字门禁->裁断->嵌字->复检），纯 CPU
  color-norm      色彩归一批量（饱和度/色相归一），纯 CPU

参数 JSON:
  repeat   每张图重复次数（放大批量用）, 默认 1
  workers  进程池大小, 默认 1
  images   指定输入文件名列表, 默认取 bench/inputs 全部
  target_sat  color-norm 目标饱和度百分比, 默认 60

产物: out/summary.json（机器信息+吞吐+逐 job 报告）、out/sample_*.png（1 张视觉验证图）
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
    ap.add_argument("--task", required=True, choices=["stamp-pipeline", "color-norm"])
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
