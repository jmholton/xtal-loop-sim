#!/usr/bin/env python
"""bench_serve.py -- how fast can THIS HOST serve frames, with no GPU at all?

The camera server replays pre-rendered templates (decode, crop/scale, camera
model, JPEG encode); this measures that CPU-only path, a different question
from how fast the box renders (`bench_frame.py`, `acceptance_voltron.py`).
No CUDA and no socket: `TemplateSource.render(pose)` is the whole serve path
below HTTP, so this runs headless on a login shell.

Three regimes: `slew` (spindle turning, a fresh decode every frame, the
worst case), `pan` (fixed angle, decode cached, only crop/scale/camera
remain), `hold` (repeated pose, the floor).  The verdict grades `slew_warm`
(a second pass, decode cache warm) when the cache can hold a full
revolution, since the server pre-warms at boot; otherwise it grades cold
`slew`.  See docs/DECISIONS.md 2026-08-14 for the crop/cache measurements
behind these numbers.

    python bench_serve.py --scene scene_files/hampton_300um_realistic.yaml
"""
import argparse
import io
import json
import os
import platform
import statistics
import sys
import time

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import numpy as np
from PIL import Image

from loop_sim.library.frame_library import (frame_for_angle, library_dir,
                                            load_manifest, pose_crop)
from loop_sim.renderer import field as _field
from loop_sim.scene.scene import load as load_scene
from loop_sim.server.camera_server import (_DECODED_BYTES_PER_PX, TemplateSource,
                                           plan_template_cache)


# The socket goal everything is graded against, shared with acceptance_voltron.
TARGET_FPS = 10.0

# Recorded BEFORE the 2026-08-14 tight crop, on `hampton_300um_realistic` with
# full-window templates, --mono on and field.to_sensor doing the 640->704
# resample.  History, not a target: they exist so a number on a login shell says
# whether anything moved without the reader cross-referencing RUNBOOK.  Keyed by
# hostname prefix because that is all a headless run knows about itself.
_BASELINES = {
    "voltron": {"label": "voltron, 2026-08-13, full-window templates",
                "slew_cold_ms": 265.5, "slew_warm_ms": 67.3,
                "cache_gib": 14.4,
                "stages": {"decode_ms": 180.7, "crop_scale_ms": 32.6,
                           "camera_model_ms": 33.0, "jpeg_encode_ms": 5.9}},
    "DESKTOP-": {"label": "dev box, 2026-08-13, full-window templates",
                 "slew_cold_ms": 91.1, "slew_warm_ms": 27.1,
                 "cache_gib": 14.4,
                 "stages": {"decode_ms": 66.6, "crop_scale_ms": 15.3,
                            "camera_model_ms": 12.3, "jpeg_encode_ms": 2.7}},
}


def _baseline_for(host):
    for prefix, rec in _BASELINES.items():
        if host.startswith(prefix):
            return rec
    return None


def _stats(samples_s):
    """median / p10 / p90 in ms, plus the implied fps, from a list of seconds.

    Median rather than mean: a single scheduler hiccup on a shared node moves
    the mean and not the median, and the question is what an operator sees
    frame after frame.
    """
    ms = sorted(1000.0 * s for s in samples_s)
    n = len(ms)
    med = statistics.median(ms)
    return {
        "frames": n,
        "median_ms": round(med, 2),
        "p10_ms": round(ms[max(0, int(0.10 * n) - 1)], 2),
        "p90_ms": round(ms[min(n - 1, int(0.90 * n))], 2),
        "fps": round(1000.0 / med, 2) if med > 0 else None,
    }


def _time_regime(src, poses, warmup_poses):
    """Wall-clock per `render()` call over `poses`, after warming on others.

    `warmup_poses` must be DISJOINT from `poses` in the slew regime, and may
    overlap in pan/hold.  Warming on the timed poses would pre-decode them
    into the 8-template cache and serve the "cold" slew warm, silently
    reading as the pan number instead.  Warmup exists to settle PIL and
    numpy, not to pre-decode the thing being measured.
    """
    for p in warmup_poses:
        src.render(p)
    out = []
    for p in poses:
        t0 = time.perf_counter()
        src.render(p)
        out.append(time.perf_counter() - t0)
    return out


def _drop_cache(src):
    """Force every later frame to decode.

    Reaches into TemplateSource's private cache on purpose: there is no public
    way to invalidate it, and a benchmark that inherited a warm cache from the
    previous regime would report the pan number for the slew.
    """
    with src._lock:
        src._cache.clear()
        src._order.clear()


def _stage_split(src, angles):  # noqa: C901
    """Cost of each stage of one frame, in ms, measured separately.

    Not a profiler: each stage is run on its own, on a cold cache, so the
    numbers add up to roughly the slew median rather than exactly.  That is
    enough to say WHERE the time goes, which is what decides whether a prefetch
    pool or a cheaper camera stage is the lever worth pulling.

    Every stage is driven through `TemplateSource`'s OWN methods rather than
    reproduced here: a split that measures a pipeline nobody runs is worse
    than no split at all.
    """
    man = src.manifest
    decode, crop, cam_stage, encode = [], [], [], []
    for ang in angles:
        rec = frame_for_angle(man, float(ang))
        box, out_size, sigma, _ = pose_crop(man, angle_deg=float(ang), zoom=1.0,
                                            clamp=True)
        _drop_cache(src)

        t0 = time.perf_counter()
        src._frame(rec["file"])         # PIL is lazy; _frame forces the decode
        decode.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        crop_im = src._compose(rec, box, out_size)
        if sigma > 0.05:
            from PIL import ImageFilter
            crop_im = crop_im.filter(ImageFilter.GaussianBlur(radius=sigma))
        crop.append(time.perf_counter() - t0)

        delivered = tuple(src.sensor) if src.sensor else out_size
        pin = src._pin(float(ang), box, delivered, sensor=None)

        # The sensor stretch is charged to the CAMERA stage, because that is
        # where `field.to_sensor` does the same work inside encode_frame.
        t0 = time.perf_counter()
        img = (src._sensor_stretch(crop_im) if src.sensor
               else np.asarray(crop_im, dtype=np.float64) / 255.0)
        if src.camera:
            img = _field.apply_camera(img, defocus=float(sigma), pin=pin,
                                      **src.camera)
        cam_stage.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        u8 = (np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(u8, mode="RGB").save(buf, format="JPEG",
                                             quality=src.jpeg_quality)
        encode.append(time.perf_counter() - t0)

    ms = lambda xs: round(1000.0 * statistics.median(xs), 2)
    return {"decode_ms": ms(decode), "crop_scale_ms": ms(crop),
            "camera_model_ms": ms(cam_stage), "jpeg_encode_ms": ms(encode)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scene", default=os.path.join(REPO, "scene_files",
                                                    "hampton_300um_realistic.yaml"))
    ap.add_argument("--root", default=None, help="library root (default: repo's)")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--slew-step", type=float, default=1.0,
                    help="degrees per frame in the slew regime (default 1.0, "
                         "which is the library's own step and guarantees a "
                         "fresh template every frame)")
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--template-cache", default="auto",
                    help="decoded templates held in RAM. auto (default) mirrors "
                         "the server's shipped default, so the numbers describe "
                         "what an operator actually gets; it is affordable now "
                         "that a template stores only its content (~1.8 GiB for "
                         "a sweep, not 14.4). 'off' is the old 8-entry "
                         "behaviour, or pass an integer")
    ap.add_argument("--no-camera", action="store_true",
                    help="serve raw transmittance -- isolates how much of the "
                         "frame is the camera model")
    ap.add_argument("--no-baseline", action="store_true",
                    help="omit the recorded pre-2026-08-14 comparison")
    ap.add_argument("--json", default=None, help="also write the report here")
    args = ap.parse_args()

    lib_dir = library_dir(args.scene, args.root) if args.root else library_dir(args.scene)
    if not os.path.exists(os.path.join(lib_dir, "manifest.json")):
        print(f"no frame library at {lib_dir} -- build one first "
              f"(python -m loop_sim.library --scene {args.scene})", file=sys.stderr)
        return 3
    man = load_manifest(lib_dir)
    scene = load_scene(args.scene, device="cpu")

    # MUST match the server's own defaults or the number describes a
    # configuration nobody runs: camera_server defaults --mono OFF (frames are
    # delivered in colour) and --pin-streak on.  mono is not free in either
    # direction -- to_luma is a whole-frame matmul plus a 3x repeat, so having
    # it ON here while the server has it off would over-report by ~1.3 ms.
    camera = None if args.no_camera else {"mono": False, "streak": True}
    sensor = tuple(_field.SENSOR_WH)
    cache = (8 if args.template_cache == "off"
             else "auto" if args.template_cache == "auto"
             else int(args.template_cache))
    src = TemplateSource(man, lib_dir, jpeg_quality=args.jpeg_quality,
                         camera=camera, sensor=sensor, scene=scene,
                         cache_size=cache)

    axis = man["axis"]
    n, step = args.frames, args.slew_step
    slew = [{axis: (i * step) % 360.0} for i in range(n)]
    # Pan holds the angle so the decode is cached, and stays inside the
    # rendered window so pose_crop does not clamp (a clamp would silently make
    # every later frame identical and flatter the number).
    pan = [{axis: 0.0, "tx": -0.25 + 0.5 * (i / max(1, n - 1))} for i in range(n)]
    hold = [{axis: 0.0}] * n

    report = {
        "schema": "loopsim-serve/1",
        "host": platform.node(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "scene": os.path.basename(args.scene),
        "library": {"frames": len(man["frames"]),
                    "rendered": man["rendered"],
                    "supersample": man["supersample"],
                    "format": man["format"]},
        "camera_emulation": camera is not None,
        "template_cache": src._cache_size,
        "template_cache_auto": plan_template_cache(man),
        "sensor": list(sensor),
    }

    # Slew warms on angles the timed run never revisits, so no timed frame can
    # be served from the 8-template cache.  Pan and hold warm on their own
    # poses, because a warm cache is the regime being measured there.
    slew_warm = [{axis: (180.0 + i * step) % 360.0} for i in range(args.warmup)]
    _drop_cache(src)
    report["slew"] = _stats(_time_regime(src, slew, slew_warm))
    # Second pass over the SAME angles, cache left warm: what a running server looks
    # like once a revolution has been walked once.  Reported separately because
    # it is only reachable when the cache can hold the sweep -- see the
    # thrash warning below for the caveat on this number.
    report["slew_warm"] = _stats(_time_regime(src, slew, []))
    _drop_cache(src)
    report["pan"] = _stats(_time_regime(src, pan, pan[:args.warmup]))
    _drop_cache(src)
    report["hold"] = _stats(_time_regime(src, hold, hold[:args.warmup]))

    _drop_cache(src)
    report["stages"] = _stage_split(
        src, [(i * step) % 360.0 for i in range(min(8, n))])

    w = report["library"]["rendered"]
    # The stored crop, not the virtual window: since templates carry only their
    # content, `rendered` over-states a decoded frame by ~9x and the resident
    # figure below would be pure fiction.
    window = int(w["width"]), int(w["height"])
    big = max((tuple(f.get("content_size_px") or window) for f in man["frames"]),
              key=lambda wh: wh[0] * wh[1])
    geom = (f"{window[0]}x{window[1]}" if big == window else
            f"{window[0]}x{window[1]} window / {big[0]}x{big[1]} stored")
    print(f"\nbench_serve -- {report['host']}  ({report['cpu_count']} cpus)")
    print(f"  scene   {report['scene']}  library {report['library']['frames']} frames "
          f"at {geom} {report['library']['format']}, "
          f"supersample {report['library']['supersample']}x")
    # 4.25 B/px, not 3: PIL stores RGB as 4-byte-aligned RGBX and the object
    # costs a little on top (measured 4.22 on real templates).  See
    # camera_server._DECODED_BYTES_PER_PX -- reporting 3 understated a
    # 360-frame sweep as 1.64 GiB when it is 2.31.
    resident = src._cache_size * big[0] * big[1] * _DECODED_BYTES_PER_PX / 2**30
    print(f"  camera emulation {'on' if camera else 'OFF'}, "
          f"delivered {sensor[0]}x{sensor[1]}")
    print(f"  template cache {src._cache_size} frames "
          f"({resident:.2f} GiB resident if fully warmed)")
    # Say it out loud rather than leaving it to be inferred from the geometry
    # line: a host that has not picked up the cropped libraries yet will read
    # ~7x slower on the decode and there is nothing in the numbers themselves to
    # explain why.
    report["library"]["cropped"] = big != window
    if big == window:
        print(f"  NOTE this library stores the FULL window -- it predates the "
              f"2026-08-14 crop.\n       Numbers below are the old regime; "
              f"`python -m loop_sim.library --recrop --all` migrates it in "
              f"minutes,\n       with no GPU and no re-render.")
    print()
    for k in ("slew", "slew_warm", "pan", "hold"):
        r = report[k]
        print(f"  {k:5s}  {r['median_ms']:7.2f} ms  ({r['fps']:6.2f} fps)   "
              f"p10 {r['p10_ms']:.2f}  p90 {r['p90_ms']:.2f}")
    s = report["stages"]
    stage_sum = sum(s.values())
    report["stages_sum_ms"] = round(stage_sum, 2)
    report["unattributed_ms"] = round(report["slew"]["median_ms"] - stage_sum, 2)
    print(f"\n  stage split (median, ms):  decode {s['decode_ms']}  "
          f"crop+scale {s['crop_scale_ms']}  camera {s['camera_model_ms']}  "
          f"jpeg {s['jpeg_encode_ms']}")
    print(f"  -> a slew pays all four; a pan skips the decode.")
    # The split is measured stage-by-stage on a handful of angles, so each
    # stage reads data the previous stage just left in cache.  A real slew
    # does not: it streams a DIFFERENT ~5 MB template through the pipeline
    # every frame, from DRAM rather than L3 (see docs/DECISIONS.md 2026-08-14,
    # voltron measured: crop lands, remaining gap is memory, for the
    # per-template-resident timings and the L3 boundary).  On a memory-bound
    # host the split can therefore under-report a slew badly, so say by how
    # much rather than let it read as measurement error.
    gap = report["unattributed_ms"]
    if gap > 0.15 * stage_sum:
        print(f"  -> the four stages sum to {stage_sum:.1f} ms but a slew "
              f"measures {report['slew']['median_ms']:.1f}: the missing "
              f"{gap:.1f} ms is per-frame\n     memory traffic the split cannot "
              f"see (it re-reads cache-hot data; a slew streams a new "
              f"{big[0] * big[1] * 3 / 1e6:.1f} MB\n     template from DRAM each "
              f"frame). Compare `pan` against `slew_warm` to see it directly.")
    n_lib = report["library"]["frames"]
    if src._cache_size < n_lib:
        print(f"\n  WARNING cache holds {src._cache_size} of {n_lib} frames. "
              f"slew_warm above is honest only for sweeps under "
              f"{src._cache_size} frames;\n          a FULL revolution evicts "
              f"each frame just before it is needed again (LRU vs a cyclic "
              f"access pattern),\n          so it will read like the cold "
              f"slew. Raise the host's RAM or --template-cache to fix.")
    else:
        print(f"\n  cache holds the whole {n_lib}-frame library: no eviction, "
              f"so a full revolution stays warm.")

    base = None if args.no_baseline else _baseline_for(report["host"])
    if base:
        report["baseline"] = base
        print(f"\n  AGAINST {base['label']}:")
        print(f"    {'':22s}{'then':>12}{'now':>12}")
        for key, label in (("slew", "slew (cold)"), ("slew_warm", "slew (warm)")):
            then = base[f"{key}_cold_ms" if key == "slew" else "slew_warm_ms"]
            now = report[key]["median_ms"]
            print(f"    {label:22s}{then:9.1f} ms{now:9.1f} ms   "
                  f"{then / max(now, 1e-9):5.2f}x  "
                  f"({1000 / then:.1f} -> {1000 / max(now, 1e-9):.1f} fps)")
        for k, label in (("decode_ms", "decode"), ("crop_scale_ms", "crop+scale"),
                         ("camera_model_ms", "camera model"),
                         ("jpeg_encode_ms", "jpeg encode")):
            then, now = base["stages"][k], s[k]
            print(f"    {label:22s}{then:9.1f} ms{now:9.1f} ms   "
                  f"{then / max(now, 1e-9):5.2f}x")
        print(f"    {'cache to do it':22s}{base['cache_gib']:9.1f} GiB"
              f"{resident:9.2f} GiB")

    # THE VERDICT IS GRADED ON THE REGIME THE VIEWER ACTUALLY SERVES.
    #
    # That is `slew_warm` whenever the cache holds a whole revolution, because
    # the server pre-warms the library at boot (`--prewarm`, default on): by
    # the time anyone drives it, every template is decoded and a spindle slew
    # never touches disk.  Grading the COLD slew there measures a state the
    # viewer only occupies during its own startup, and it is also the noisy
    # one (first-touch I/O off a shared pool; see docs/DECISIONS.md
    # 2026-08-14, voltron measured: crop lands, remaining gap is memory).
    #
    # When the cache CANNOT hold a revolution, pre-warm is skipped and every
    # rotating frame really does decode, so cold is the grade.
    warm_reachable = src._cache_size >= report["library"]["frames"]
    graded_key = "slew_warm" if warm_reachable else "slew"
    graded = report[graded_key]["fps"]
    cold = report["slew"]["fps"]
    report["target_fps"] = TARGET_FPS
    report["graded_regime"] = graded_key
    report["verdict"] = "GO" if graded >= TARGET_FPS else "NO-GO"
    what = ("a slew on the pre-warmed library serves at" if warm_reachable
            else "this host cannot cache a revolution, so a slew serves at")
    print(f"\n  VERDICT  {report['verdict']}: {what} {graded:.2f} fps "
          f"against the {TARGET_FPS:g} fps goal"
          + ("" if graded >= TARGET_FPS else " -- not usable as a viewer"))
    print(f"           (a browser shows roughly half the socket rate, so "
          f"~{graded / 2:.1f} fps is what an operator sees)")
    if warm_reachable:
        print(f"           the {cold:.2f} fps cold figure above is the "
              f"benchmark decoding from scratch; the server pays that once at\n"
              f"           boot instead (--prewarm), so it is startup cost, not "
              f"something an operator meets.")
    print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"  wrote {args.json}")
    return 0 if graded >= TARGET_FPS else 1


if __name__ == "__main__":
    sys.exit(main())
