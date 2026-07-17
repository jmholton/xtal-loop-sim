#!/usr/bin/env python
"""
acceptance_voltron.py -- self-contained performance acceptance test for loop-sim
on its deployment GPU (e.g. a free voltron TITAN V).

It exists because the 10 fps interactive target was only ever measured on a
developer RTX 4080 SUPER, never on the beamline's TITAN V.  Run this ON the
deployment machine to get the real answer; no Nsight, no extra installs, no dev
access -- it drives the repo's own engine.

WHAT IT ANSWERS
  1. Does the 10 fps target hold on THIS machine's GPU + CPU?  (the frame is
     CPU-dispatch-bound, so the host CPU matters as much as the GPU.)
  2. Does torch.compile actually ENGAGE -- the whole basis of 10 fps -- or does
     it silently fall back to eager (~6-7 fps)?               [deploy Risk B]
  3. Does the mesh scene (mitegen_200um) fit in VRAM, or OOM?  [deploy Risk A]
  4. A full environment fingerprint (GPU, driver, CUDA, torch, CPU) so a run
     here is directly comparable to a run on the dev box.

HOW TO RUN
    /programs/pytorch/envs/pt/bin/python acceptance_voltron.py
  On a shared multi-GPU box it auto-selects the GPU with the most free memory
  (voltron has 8 TITAN Vs and is usually busy -- pin explicitly with --gpu N or
  CUDA_VISIBLE_DEVICES if you prefer).  Writes acceptance_report.json and prints
  a GO / NO-GO verdict.  Mail the JSON back for comparison with the prediction.

EXIT CODE   0 = GO (>= 10 fps)   1 = NO-GO (< 10 fps)   3 = could not run (no CUDA)
"""
import argparse
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import time

TARGET_FPS = 10.0
REPO = os.path.dirname(os.path.abspath(__file__))

# The 10 fps path is torch.compile (torch >= 2.0 -> Python >= 3.9). The legacy
# system python (2.7 / 3.6 on the beamline) cannot run it. Fail early and clearly
# rather than crash cryptically, and point at the interpreter the GPU jobs use.
if sys.version_info < (3, 9):
    sys.stderr.write(
        "\nloop-sim acceptance needs Python >= 3.9 with torch 2.x (torch.compile is\n"
        "the 10 fps path). You are on Python %s.\n"
        "On voltron use the python3.10 interpreter the GPU jobs already run, e.g.:\n"
        "  /programs/pytorch/envs/pt/bin/python3.10 %s\n\n"
        % (platform.python_version(), sys.argv[0]))
    raise SystemExit(3)


def _smi(query):
    """Return nvidia-smi --query-gpu rows as a list of strings (empty on failure)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=" + query, "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL)
        return [r.strip() for r in out.strip().splitlines()]
    except Exception:
        return []


def pick_freest_gpu():
    """(index, free_MiB) of the most-free GPU per nvidia-smi, or None."""
    best = None
    for row in _smi("index,memory.free"):
        try:
            idx, free = [x.strip() for x in row.split(",")]
            idx, free = int(idx), int(free)
        except Exception:
            continue
        if best is None or free > best[1]:
            best = (idx, free)
    return best


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def _p90(xs):
    xs = [x for x in xs if x == x]  # drop NaN
    if len(xs) < 2:
        return float(xs[0]) if xs else float("nan")
    return statistics.quantiles(xs, n=10)[8]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, default=None,
                    help="GPU index to use (default: auto-pick the most-free GPU)")
    ap.add_argument("--frames", type=int, default=30, help="timed frames per measurement")
    ap.add_argument("--warmup", type=int, default=8,
                    help="warmup frames (torch.compile settles here)")
    ap.add_argument("--out", default=os.path.join(REPO, "acceptance_report.json"))
    ap.add_argument("--no-mesh", action="store_true",
                    help="skip the mitegen mesh VRAM test (the OOM-risk scene)")
    args = ap.parse_args()

    # --- pin the GPU BEFORE importing torch (so a shared box isn't grabbed at random) ---
    chosen, chosen_free = args.gpu, None
    if chosen is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen)
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        pick = pick_freest_gpu()
        if pick:
            chosen, chosen_free = pick
            os.environ["CUDA_VISIBLE_DEVICES"] = str(chosen)

    driver = (_smi("driver_version") or ["unknown"])[0]

    report = {
        "schema": "loopsim-acceptance/1",
        "host": socket.gethostname(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": cpu_model(),
        "cpu_count": os.cpu_count(),
        "gpu_index_used": chosen,
        "gpu_free_at_pick_mib": chosen_free,
        "driver_version": driver,
        "target_fps": TARGET_FPS,
    }

    import torch  # noqa: E402  (imported after CUDA_VISIBLE_DEVICES is set)
    report["torch"] = torch.__version__
    report["torch_cuda_build"] = torch.version.cuda

    if not torch.cuda.is_available():
        report["fatal"] = "torch.cuda.is_available() is False"
        _emit(report, args.out)
        print("\nFATAL: no usable CUDA device -- cannot run the acceptance test.")
        print("  torch %s (cuda build %s) on driver %s" % (torch.__version__, torch.version.cuda, driver))
        return 3

    try:
        import torch._dynamo as dynamo
        _has_compile = hasattr(torch, "compile")
    except Exception:
        dynamo, _has_compile = None, False
    from loop_sim.scene.scene import load
    from loop_sim.motors.goniometer import Goniometer
    from loop_sim.renderer.engine_torch import TorchScene, render_torch

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    report["gpu_name"] = props.name
    report["gpu_cc"] = "%d.%d" % (props.major, props.minor)
    report["gpu_total_mib"] = round(props.total_memory / 2**20, 1)
    report["is_titan_v"] = "TITAN V" in props.name.upper()

    def build(scene_name):
        return TorchScene(load(os.path.join(REPO, "scene_files", scene_name)), dev, torch.float64)

    def moving_poses(ts, n):
        # A /motor stream: a fresh centred-loop pose each frame (the worst case for
        # the AXIS consumer). The small rot sweep also varies the survivor batch,
        # which stresses the dynamic-shape compiled path the way real motion does.
        return [Goniometer(ts.scene.geometry).set(rotx=float(i % 12)) for i in range(max(1, n))]

    def time_path(ts, compiled, n_cond, frames, warmup):
        poses = moving_poses(ts, max(frames, warmup))
        for i in range(warmup):
            render_torch(ts, poses[i], n_cond=n_cond, compiled=compiled)
        torch.cuda.synchronize()
        out = []
        for i in range(frames):
            t0 = time.perf_counter()
            render_torch(ts, poses[i], n_cond=n_cond, compiled=compiled)
            torch.cuda.synchronize()
            out.append((time.perf_counter() - t0) * 1e3)
        return statistics.median(out), min(out), _p90(out)

    ham = build("hampton_300um.yaml")

    # --- 1. eager baseline (the fallback floor) ---
    eager_med, eager_min, eager_p90 = time_path(ham, False, 1, args.frames, args.warmup)

    # --- 2. compiled preview path (the 10 fps lever). Force compile errors to
    #        SURFACE instead of silently degrading -- that silent fallback is Risk B. ---
    compile_error = None
    frames_compiled = 0
    if not _has_compile:
        # torch < 2.0 (e.g. the legacy py3.6/torch-1.x env) has no torch.compile at
        # all -> the 10 fps preview path cannot exist here. Report it, skip cleanly.
        compile_error = "torch %s has no torch.compile (need >= 2.0; use the python3.10 env)" % torch.__version__
        comp_med = comp_min = comp_p90 = float("nan")
    else:
        dynamo.reset()
        prev_suppress = dynamo.config.suppress_errors
        dynamo.config.suppress_errors = False
        try:
            comp_med, comp_min, comp_p90 = time_path(ham, True, 1, args.frames, args.warmup)
        except Exception as e:  # Volta codegen / Inductor failure lands here
            compile_error = "%s: %s" % (type(e).__name__, str(e).splitlines()[0] if str(e) else "")
            comp_med = comp_min = comp_p90 = float("nan")
        finally:
            dynamo.config.suppress_errors = prev_suppress
        frames_compiled = dynamo.utils.counters.get("frames", {}).get("ok", 0)
    # Operative signal is a REAL speedup with no error -- robust across torch
    # versions (the dynamo counter keys changed between 2.0 and 2.6, so treat the
    # graph count as corroborating only, never as the gate).
    compiled_engaged = frames_compiled > 0
    compiled_faster = compile_error is None and comp_med == comp_med and comp_med < eager_med * 0.9
    compiled_ok = bool(compiled_faster)

    # --- 3. settle frame (exact n_cond=7 eager -- the still-image quality path) ---
    settle_med, _, _ = time_path(ham, False, 7, max(4, args.frames // 5), 3)

    # deployed motion path = compiled if it truly engaged+helped, else the eager floor
    deployed_ms = comp_med if compiled_ok else eager_med
    deployed_fps = 1000.0 / deployed_ms

    report["fps"] = {
        "eager_preview_ms": round(eager_med, 2),
        "eager_preview_fps": round(1000.0 / eager_med, 2),
        "compiled_preview_ms": None if comp_med != comp_med else round(comp_med, 2),
        "compiled_preview_fps": None if comp_med != comp_med else round(1000.0 / comp_med, 2),
        "compiled_p90_ms": None if comp_p90 != comp_p90 else round(comp_p90, 2),
        "settle_n7_ms": round(settle_med, 1),
        "deployed_motion_fps": round(deployed_fps, 2),
    }
    report["compile"] = {
        "engaged": compiled_engaged,
        "frames_compiled": frames_compiled,
        "faster_than_eager": compiled_faster,
        "compiled_ok": compiled_ok,
        "error": compile_error,
    }

    # --- 4. VRAM: hampton (tube, small) then mitegen (mesh, the OOM risk) ---
    def vram_probe(scene_name, n_cond):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        oom = None
        try:
            ts = build(scene_name)
            render_torch(ts, Goniometer(ts.scene.geometry).set(), n_cond=n_cond, compiled=False)
            torch.cuda.synchronize()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                oom = str(e).splitlines()[0]
            else:
                raise
        peak = torch.cuda.max_memory_reserved() / 2**20
        torch.cuda.empty_cache()
        return round(peak, 1), oom

    ham_peak, _ = vram_probe("hampton_300um.yaml", 1)
    report["vram"] = {"hampton_peak_reserved_mib": ham_peak, "total_mib": report["gpu_total_mib"]}
    if args.no_mesh:
        report["vram"]["mitegen"] = "skipped"
        mesh_ok = None
    else:
        mesh_peak, mesh_oom = vram_probe("mitegen_200um.yaml", 1)
        report["vram"]["mitegen_peak_reserved_mib"] = mesh_peak
        report["vram"]["mitegen_oom"] = mesh_oom
        mesh_ok = mesh_oom is None
        report["vram"]["mitegen_headroom_mib"] = (
            None if mesh_oom else round(report["gpu_total_mib"] - mesh_peak, 1))

    # --- verdict ---
    fps_go = deployed_fps >= TARGET_FPS
    report["verdict"] = {
        "fps": "GO" if fps_go else "NO-GO",
        "compile": ("OK" if compiled_ok
                    else "FAILED" if compile_error
                    else ("COMPILED-NO-SPEEDUP" if compiled_engaged else "SILENT-FALLBACK")),
        "mesh_vram": ("SKIPPED" if mesh_ok is None else ("FITS" if mesh_ok else "OOM")),
        "overall": "GO" if fps_go else "NO-GO",
    }

    _emit(report, args.out)
    _print_summary(report, args.out)
    return 0 if fps_go else 1


def _emit(report, out):
    with open(out, "w") as f:
        json.dump(report, f, indent=2)


def _print_summary(r, out):
    v, fps, comp, vram = r["verdict"], r["fps"], r["compile"], r["vram"]
    line = "=" * 68
    print("\n" + line)
    print(" loop-sim acceptance  --  %s" % r.get("gpu_name", "?"))
    print(line)
    print("  host        : %s" % r["host"])
    print("  GPU         : %s (cc %s, %.0f MiB, driver %s)" % (
        r.get("gpu_name", "?"), r.get("gpu_cc", "?"), r.get("gpu_total_mib", 0), r["driver_version"]))
    print("  CPU         : %s (%s threads)" % (r["cpu"], r["cpu_count"]))
    print("  torch       : %s  (cuda build %s)" % (r["torch"], r["torch_cuda_build"]))
    print("  " + "-" * 64)
    print("  eager preview   : %6.1f ms  = %5.1f fps" % (fps["eager_preview_ms"], fps["eager_preview_fps"]))
    if fps["compiled_preview_ms"] is not None:
        print("  compiled preview: %6.1f ms  = %5.1f fps   (p90 %.1f ms)" % (
            fps["compiled_preview_ms"], fps["compiled_preview_fps"], fps["compiled_p90_ms"]))
    else:
        print("  compiled preview: FAILED -- %s" % comp["error"])
    print("  settle (n=7)    : %6.1f ms" % fps["settle_n7_ms"])
    print("  " + "-" * 64)
    print("  compile engaged : %s   (%d graphs; %s)" % (
        comp["engaged"], comp["frames_compiled"], v["compile"]))
    if "mitegen_peak_reserved_mib" in vram:
        if vram.get("mitegen_oom"):
            print("  mesh VRAM       : OOM at %.0f MiB total  -> %s" % (vram["total_mib"], vram["mitegen_oom"]))
        else:
            print("  mesh VRAM       : peak %.0f / %.0f MiB  (headroom %.0f MiB)  -> FITS" % (
                vram["mitegen_peak_reserved_mib"], vram["total_mib"], vram["mitegen_headroom_mib"]))
    print(line)
    print("  DEPLOYED MOTION : %.1f fps   (target %.0f)" % (fps["deployed_motion_fps"], r["target_fps"]))
    print("  VERDICT         : %s" % v["overall"])
    print(line)
    print("  wrote %s" % os.path.abspath(out))


if __name__ == "__main__":
    sys.exit(main())
