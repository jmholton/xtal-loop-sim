#!/usr/bin/env python
"""bench_serve.py -- how fast can THIS HOST serve frames, with no GPU at all?

The camera server replays pre-rendered templates: it decodes a PNG, crops and
scales it to the pose, blurs it by the defocus, and runs the camera model.  No
ray tracing, no CUDA.  So the frame rate an operator sees is a property of the
HOST CPU, and it is a different question from "how fast does this box render",
which `bench_frame.py` and `acceptance_voltron.py` already answer.

It matters because the two can point in opposite directions.  A machine with a
strong GPU and an old CPU -- voltron is 8x TITAN V in front of a 2016 Xeon
E5-2650 v4 -- may build libraries faster than the dev box and still serve them
slower.  Nothing in the repo measured that until this file.

WHAT IT REPORTS, AND WHY THE SPLIT IS THE POINT

  slew   the spindle turning: every frame is a different template, so every
         frame pays a full PNG decode.  This is the worst case and the one
         that sets the frame rate an operator perceives while rotating.
  pan    translating at a fixed angle: the decoded template is reused from
         `TemplateSource`'s cache, so the decode disappears and only the
         crop/scale/camera stages remain.  Measured on the dev box this is
         ~5x cheaper than a slew, so reporting one number for "serving" would
         hide the entire effect.
  hold   the same pose repeatedly.  The live server would answer this from its
         own JPEG cache without re-rendering at all; it is here as the floor,
         to separate fixed overhead from real work.

Then a stage split -- decode / crop+scale / camera model / JPEG encode --
because the remedy differs per stage.  A prefetch decode pool is already the
identified lever for the slew case (~30 fps, no library rebuild); it would do
nothing for the camera stage.

DELIBERATELY NO SOCKET.  `TemplateSource.render(pose)` is the whole serve path
below HTTP and returns the JPEG bytes, so this runs headless on a login shell
with no port to bind and no browser -- which is the only way to benchmark a
shared beamline node.

THE CACHE IS LOAD-BEARING, AND IT IS NOW THE THING UNDER TEST.  `TemplateSource`
sizes its decode cache from available RAM (`plan_template_cache`), so on a box
that can hold the library the slew SHOULD collapse onto the pan number once
warm -- that is the whole point of the 2026-08-13 change.  This benchmark
deliberately drops the cache between regimes and warms the slew on angles the
timed run never revisits, so what it reports is the COLD cost of each regime.
Use `--template-cache 8` to see the old behaviour and `--frames` larger than the
cache to keep measuring cold decodes on a big-cache host.

USAGE
    python bench_serve.py --scene scene_files/hampton_300um_realistic.yaml
    python bench_serve.py --scene ... --frames 60 --json serve_report.json
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
from loop_sim.motors.goniometer import Goniometer
from loop_sim.renderer import field as _field
from loop_sim.renderer.pin_projection import project_pin, template_mapper
from loop_sim.scene.scene import load as load_scene
from loop_sim.server.camera_server import (TemplateSource, encode_frame,
                                           plan_template_cache, pose_phase)


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
    overlap in pan/hold.  Warming on the timed poses themselves was the first
    version of this function and it silently lied: the decode cache holds 8
    templates, so the first frames of the timed run were served warm and the
    slew median came out 78 ms with a p10 of 25.8 -- the p10 being, exactly,
    the pan number.  Warmup exists to settle PIL and numpy, not to pre-decode
    the thing being measured.
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


def _stage_split(man, lib_dir, scene, camera, sensor, angles):  # noqa: C901
    """Cost of each stage of one frame, in ms, measured separately.

    Not a profiler: each stage is run on its own, on a cold cache, so the
    numbers add up to roughly the slew median rather than exactly.  That is
    enough to say WHERE the time goes, which is what decides whether a prefetch
    pool or a cheaper camera stage is the lever worth pulling.
    """
    decode, crop, cam_stage, encode = [], [], [], []
    for ang in angles:
        rec = frame_for_angle(man, float(ang))
        box, out_size, sigma, _ = pose_crop(man, angle_deg=float(ang), zoom=1.0,
                                            clamp=True)
        path = os.path.join(lib_dir, rec["file"])

        t0 = time.perf_counter()
        im = Image.open(path).convert("RGB")
        im.load()                       # PIL is lazy; load() is the real decode
        decode.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        crop_im = im.resize(out_size, Image.BILINEAR, box=box)
        if sigma > 0.05:
            from PIL import ImageFilter
            crop_im = crop_im.filter(ImageFilter.GaussianBlur(radius=sigma))
        arr = np.asarray(crop_im, dtype=np.float64) / 255.0
        crop.append(time.perf_counter() - t0)

        # The pin is projected exactly as TemplateSource._pin does it.  An
        # earlier version passed pin=None here, which quietly dropped the
        # specular glint from the camera stage and under-counted it -- the
        # timed regimes above always drew it, so the split did not add up.
        to_px, frame_wh = template_mapper(man, box, out_size, sensor)
        gono = Goniometer(scene.geometry).set(**{man["axis"]: float(ang)})
        pin = project_pin(scene, gono, to_px, frame_wh)

        t0 = time.perf_counter()
        img = _field.to_sensor(arr, sensor) if sensor else arr
        if camera:
            img = _field.apply_camera(img, defocus=float(sigma), pin=pin,
                                      **camera)
        cam_stage.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        u8 = (np.clip(np.asarray(img, dtype=np.float64), 0.0, 1.0) * 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(u8, mode="RGB").save(buf, format="JPEG", quality=85)
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
    ap.add_argument("--template-cache", default="off",
                    help="decoded templates held in RAM. off (default, = 8) "
                         "matches the server's shipped default, so the numbers "
                         "describe what an operator actually gets. 'auto' "
                         "sizes from available memory, or pass an integer -- "
                         "use those to see what opting in would buy")
    ap.add_argument("--no-camera", action="store_true",
                    help="serve raw transmittance -- isolates how much of the "
                         "frame is the camera model")
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
    # configuration nobody runs: camera_server defaults --mono on and
    # --pin-streak on.  mono is not free -- to_luma is a whole-frame matmul
    # plus a 3x repeat -- so benching with it off silently under-reports.
    camera = None if args.no_camera else {"mono": True, "streak": True}
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
    # Second pass over the SAME angles, cache left warm: what production looks
    # like once a revolution has been walked once.  Reported separately because
    # it is only reachable when the cache can hold the sweep -- see the
    # thrash warning below, which is the honest caveat on this number.
    report["slew_warm"] = _stats(_time_regime(src, slew, []))
    _drop_cache(src)
    report["pan"] = _stats(_time_regime(src, pan, pan[:args.warmup]))
    _drop_cache(src)
    report["hold"] = _stats(_time_regime(src, hold, hold[:args.warmup]))

    _drop_cache(src)
    report["stages"] = _stage_split(man, lib_dir, scene, camera, sensor,
                                    [(i * step) % 360.0 for i in range(min(8, n))])

    w = report["library"]["rendered"]
    print(f"\nbench_serve -- {report['host']}  ({report['cpu_count']} cpus)")
    print(f"  scene   {report['scene']}  library {report['library']['frames']} frames "
          f"at {w['width']}x{w['height']} {report['library']['format']}, "
          f"supersample {report['library']['supersample']}x")
    resident = src._cache_size * w["width"] * w["height"] * 3 / 2**30
    print(f"  camera emulation {'on' if camera else 'OFF'}, "
          f"delivered {sensor[0]}x{sensor[1]}")
    print(f"  template cache {src._cache_size} frames "
          f"({resident:.1f} GiB resident if fully warmed)\n")
    for k in ("slew", "slew_warm", "pan", "hold"):
        r = report[k]
        print(f"  {k:5s}  {r['median_ms']:7.2f} ms  ({r['fps']:6.2f} fps)   "
              f"p10 {r['p10_ms']:.2f}  p90 {r['p90_ms']:.2f}")
    s = report["stages"]
    print(f"\n  stage split (median, ms):  decode {s['decode_ms']}  "
          f"crop+scale {s['crop_scale_ms']}  camera {s['camera_model_ms']}  "
          f"jpeg {s['jpeg_encode_ms']}")
    print(f"  -> a slew pays all four; a pan skips the decode.")
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
    print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2)
        print(f"  wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
