#!/usr/bin/env python
"""
Warm-frame benchmark harness for the GPU-resident engine (engine_torch).

Times render_torch across a scripted pose set at n_cond=1 and 7, and reports:
  * median / p10 / p90 wall ms per frame (torch.cuda.synchronize-bracketed)
  * the server-side encode tail (uint8 conversion + D2H + PIL JPEG q85)
  * torch op-invocation count for one frame (CPU profiler; CUPTI kernel
    timings are unavailable under WSL2, so op count is the dispatch proxy)
  * GPU utilisation sampled via nvidia-smi during the timed loop
  * peak CUDA memory (watch the WSL2 ~15.5 GB spill cliff at high zoom)

Every run writes a JSON record to bench_results/ so phases can be compared:

    ~/miniconda3/envs/loopsim/bin/python bench_frame.py --label baseline
    ~/miniconda3/envs/loopsim/bin/python bench_frame.py --quick
"""
import argparse
import io
import json
import os
import statistics
import subprocess
import sys
import threading
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
from PIL import Image

from loop_sim.scene.scene import load
from loop_sim.motors.goniometer import Goniometer
from loop_sim.renderer.engine_torch import TorchScene, render_torch

POSES = {
    "id":     {},
    "rotx45": {"rotx": 45},
    "zoom2":  {"zoom": 2.0},
    "mix":    {"rotx": 33, "tx": 0.07},
    "zoom4":  {"zoom": 4.0},
}


class GpuSampler:
    """Poll nvidia-smi in a thread while a timed loop runs."""

    def __init__(self, interval=0.25):
        self.interval = interval
        self.util = []
        self.mem_mb = []
        self._stop = threading.Event()
        self._thread = None

    def _poll(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout.strip()
                u, m = out.split(",")
                self.util.append(float(u))
                self.mem_mb.append(float(m))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


def encode_jpeg(img, quality=85):
    """The camera_server encode tail: clamp -> uint8 -> D2H -> PIL JPEG."""
    img8 = (img * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
    buf = io.BytesIO()
    Image.fromarray(img8, mode="RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def op_count_one_frame(tscene, gono, n_cond, compiled=False):
    """Torch op invocations for one frame (dispatch-boundness proxy)."""
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        render_torch(tscene, gono, n_cond=n_cond, compiled=compiled)
        if tscene.dev.type == "cuda":
            torch.cuda.synchronize()
    ka = prof.key_averages()
    return int(sum(evt.count for evt in ka)), float(
        sum(evt.self_cpu_time_total for evt in ka) / 1000.0)


def bench_config(tscene, pose, n_cond, frames, warmup, compiled=False):
    gono = Goniometer(tscene.scene.geometry).set(**POSES[pose])
    cuda = tscene.dev.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    # First compiled call triggers a ~30-60 s compilation; warm up until stable.
    for _ in range(warmup):
        img = render_torch(tscene, gono, n_cond=n_cond, compiled=compiled)
    if cuda:
        torch.cuda.synchronize()

    times = []
    with GpuSampler() as smp:
        for _ in range(frames):
            t0 = time.perf_counter()
            img = render_torch(tscene, gono, n_cond=n_cond, compiled=compiled)
            if cuda:
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)

    t0 = time.perf_counter()
    encode_jpeg(img)
    encode_ms = (time.perf_counter() - t0) * 1000.0

    ops, self_cpu_ms = op_count_one_frame(tscene, gono, n_cond, compiled=compiled)
    med = statistics.median(times)
    q = statistics.quantiles(times, n=10) if len(times) >= 10 else [min(times)] * 9
    return {
        "pose": pose,
        "n_cond": n_cond,
        "compiled": compiled,
        "frames": frames,
        "median_ms": round(med, 2),
        "p10_ms": round(q[0], 2),
        "p90_ms": round(q[8], 2),
        "fps": round(1000.0 / med, 2),
        "encode_ms": round(encode_ms, 2),
        "ops_per_frame": ops,
        "profiler_self_cpu_ms": round(self_cpu_ms, 1),
        "gpu_util_mean": round(statistics.mean(smp.util), 1) if smp.util else None,
        "gpu_util_max": max(smp.util) if smp.util else None,
        "gpu_mem_used_max_mb": max(smp.mem_mb) if smp.mem_mb else None,
        "torch_peak_alloc_mb": round(
            torch.cuda.max_memory_allocated() / 2**20, 1) if cuda else None,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--scene", default=os.path.join(REPO_ROOT, "scene_files",
                                                    "hampton_300um.yaml"))
    ap.add_argument("--n-cond", default="1,7")
    ap.add_argument("--poses", default="id,rotx45,zoom2,mix,zoom4")
    ap.add_argument("--frames", type=int, default=None,
                    help="frames per config (default: 30 at n_cond=1, 8 above)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--label", default="run")
    ap.add_argument("--quick", action="store_true",
                    help="id pose only, 10 frames, n_cond=1")
    ap.add_argument("--compiled", action="store_true",
                    help="bench the torch.compile()d preview path (CUDA only): "
                         "warms up (first call compiles ~30-60 s) then times "
                         "render_torch(compiled=True)")
    ap.add_argument("--fp32", action="store_true",
                    help="bench a float32 TorchScene (the preview-scene dtype; "
                         "tube/mesh intersection math stays float64 internally)")
    args = ap.parse_args()

    if args.quick:
        args.poses, args.n_cond, args.frames = "id", "1", 10

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if dev.type != "cuda":
        print("WARNING: CUDA unavailable — timings will not match the GPU baseline.")
    compiled = args.compiled and dev.type == "cuda"
    if args.compiled and not compiled:
        print("WARNING: --compiled ignored (compilation is CUDA-only here).")
    scene = load(args.scene, device="cpu")
    dt = torch.float32 if args.fp32 else torch.float64
    tscene = TorchScene(scene, dev, dt)

    results = []
    for n_cond in [int(x) for x in args.n_cond.split(",")]:
        frames = args.frames or (30 if n_cond == 1 else 8)
        for pose in args.poses.split(","):
            r = bench_config(tscene, pose, n_cond, frames, args.warmup, compiled=compiled)
            results.append(r)
            tag = "C" if compiled else " "
            print(f"[{tag}] n_cond={n_cond} pose={pose:<7} median={r['median_ms']:8.2f} ms "
                  f"({r['fps']:5.2f} fps)  p90={r['p90_ms']:8.2f}  "
                  f"encode={r['encode_ms']:5.2f} ms  ops={r['ops_per_frame']:5d}  "
                  f"util={r['gpu_util_mean']}%  peak={r['torch_peak_alloc_mb']} MB")

    if compiled:
        try:
            import torch._dynamo as _dyn
            stats = dict(_dyn.utils.counters.get("stats", {}))
            print(f"dynamo stats (recompile watch): {stats}")
        except Exception:
            pass

    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                         capture_output=True, text=True).stdout.strip() or "nogit"
    record = {
        "label": args.label,
        "commit": sha,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu",
        "scene": os.path.basename(args.scene),
        "compiled": compiled,
        "results": results,
    }
    outdir = os.path.join(REPO_ROOT, "bench_results")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{time.strftime('%Y%m%d_%H%M%S')}_{sha}_{args.label}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nwrote {os.path.relpath(path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
