"""CLI: build (or refresh) a pre-computed frame library.

    python -m loop_sim.library --scene scene_files/hampton_300um.yaml
    python -m loop_sim.library --all            # every scene in scene_files/
    python -m loop_sim.library --scene X --force

Libraries land in frame_library/<scene_stem>/ and are tracked in git -- they
are part of the deliverable, not build output.

Tile size defaults to `auto`, which calibrates peak memory against two cheap
probe renders and sizes each pass to fit inside --vram-fraction of free VRAM.

--modality xray builds the X-ray radiograph library instead (16-bit
greyscale, no --n-cond/--psf/--quality/--format -- the tracer has none of
those; --tile-size/--recrop/--preview are also optical-only and not
supported here yet). Separate root (xray_library/) and render_sha -- see
loop_sim/library/xray_library.py and docs/DECISIONS.md 2026-08-18.

    python -m loop_sim.library --modality xray --scene scene_files/hampton_300um.yaml
"""
import argparse
import glob
import os
import sys

from .frame_library import (CPU_BUILD_REFUSAL, DEFAULT_FORMAT, DEFAULT_N_COND,
                            DEFAULT_PAN_MM, DEFAULT_PREVIEW_ROOT,
                            DEFAULT_QUALITY, DEFAULT_ROOT, DEFAULT_STEP_DEG,
                            DEFAULT_SUPERSAMPLE, DEFAULT_VRAM_FRACTION,
                            PREVIEW_BUILD, build_library, build_params,
                            cuda_available, is_current, library_dir,
                            recrop_library, zoom_limits)
from .xray_library import DEFAULT_ROOT as XRAY_DEFAULT_ROOT
from .xray_library import (build_xray_library, xray_build_params,
                           xray_is_current)
from .xray_library import zoom_limits as xray_zoom_limits


def _main_xray(args, scenes):
    """The --modality xray build loop -- see this module's docstring.

    No --n-cond/--psf/--quality/--format/--tile-size/--vram-fraction: the
    tracer has none of those knobs. Kept as a separate function rather than
    threading a branch through the optical loop below, since the two share
    almost no options.
    """
    root = XRAY_DEFAULT_ROOT if args.root == DEFAULT_ROOT else args.root
    opts = dict(axis=args.axis, step_deg=args.step, supersample=args.supersample,
               pan_mm=args.pan_mm)
    refuse_cpu = (not args.allow_cpu) and (
        args.device == "cpu" or (args.device is None and not cuda_available()))

    rc = 0
    for s in scenes:
        lib = library_dir(s, root)
        if not args.force and xray_is_current(s, lib, **xray_build_params(**opts)):
            print(f"[xray-library] {s}: already current -> {lib}")
            continue
        if refuse_cpu:
            print(f"[xray-library] {s}: {CPU_BUILD_REFUSAL}", file=sys.stderr)
            rc = 2
            continue
        print(f"[xray-library] building {s} -> {lib}")
        try:
            man = build_xray_library(s, root=root, device=args.device, **opts)
        except RuntimeError as exc:
            print(f"[xray-library] FAILED {s}: {exc}", file=sys.stderr)
            rc = 1
            continue
        n = len(man["frames"])
        size = sum(os.path.getsize(os.path.join(lib, f["file"]))
                   for f in man["frames"]) / 2**20
        rnd = man["rendered"]
        zmin, zmax = xray_zoom_limits(man)
        print(f"[xray-library] {s}: {n} frames at {rnd['width']}x{rnd['height']}, "
              f"{size:.1f} MB, zoom range {zmin:.2f}..{zmax:.0f}x")
    return rc


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m loop_sim.library",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", action="append", default=[],
                   help="scene YAML to build (repeatable)")
    p.add_argument("--all", action="store_true",
                   help="build every scene_files/*.yaml")
    p.add_argument("--root", default=DEFAULT_ROOT, help="library root dir")
    p.add_argument("--axis", default="rotx", help="spindle motor (default rotx)")
    p.add_argument("--step", type=float, default=DEFAULT_STEP_DEG,
                   help=f"degrees between frames (default {DEFAULT_STEP_DEG})")
    p.add_argument("--supersample", type=int, default=DEFAULT_SUPERSAMPLE,
                   help="render this many times finer than the camera pixel "
                        "size; the hard ceiling on zoom-in "
                        f"(default {DEFAULT_SUPERSAMPLE})")
    p.add_argument("--pan-mm", type=float, default=DEFAULT_PAN_MM,
                   help="sample travel to allow beyond the scene and the "
                        f"centred field of view (default {DEFAULT_PAN_MM} mm)")
    p.add_argument("--n-cond", type=int, default=DEFAULT_N_COND)
    p.add_argument("--quality", type=int, default=DEFAULT_QUALITY,
                   help="JPEG quality of the stored templates; ignored when "
                        "--format is png (the default)")
    p.add_argument("--format", choices=["png", "jpeg"], default=DEFAULT_FORMAT,
                   help=f"stored template format (default {DEFAULT_FORMAT}). "
                        "png is lossless and, for these near-binary frames, "
                        "also smaller than jpeg")
    p.add_argument("--psf", choices=["on", "off"], default="on",
                   help="on (default): bake the objective's diffraction PSF "
                        "into the templates. off reproduces the purely "
                        "geometric pre-2026-08 output")
    p.add_argument("--tile-size", default="auto",
                   help="rays per trace pass, or 'auto' to size from free VRAM")
    p.add_argument("--vram-fraction", type=float, default=DEFAULT_VRAM_FRACTION,
                   help="fraction of free VRAM the auto tile may use "
                        f"(default {DEFAULT_VRAM_FRACTION})")
    p.add_argument("--device", default=None,
                   help="cuda or cpu (default: cuda when available)")
    p.add_argument("--force", action="store_true", help="rebuild even if current")
    p.add_argument("--preview", action="store_true",
                   help="build the same coarse library the camera server builds "
                        "when you switch it to an unbuilt scene "
                        f"({PREVIEW_BUILD['step_deg']:g}deg steps, "
                        f"{PREVIEW_BUILD['supersample']}x supersample, "
                        f"n_cond {PREVIEW_BUILD['n_cond']}), into "
                        "frame_library_preview/. Minutes instead of the best "
                        "part of an hour; zoom is capped at 1x")
    p.add_argument("--recrop", action="store_true",
                   help="crop an EXISTING library's frames down to their "
                        "content, in place. A migration, not a build: no GPU, "
                        "no re-render, minutes rather than hours, and the "
                        "surviving pixels are bit-identical. Skips libraries "
                        "already cropped")
    p.add_argument("--allow-cpu", action="store_true",
                   help="build even with no CUDA device. A CPU build runs at "
                        "roughly 179 s/frame -- ~3.6 h for a 72-frame preview "
                        "and ~18 h for a full library -- so it is refused "
                        "unless you ask for it explicitly")
    p.add_argument("--modality", choices=["optical", "xray"], default="optical",
                   help="optical (default): the bright-field camera library. "
                        "xray: the transmission-radiograph library -- see "
                        "this module's docstring")
    args = p.parse_args(argv)

    scenes = list(args.scene)
    if args.all:
        scenes += sorted(glob.glob(os.path.join("scene_files", "*.yaml")))
    if not scenes:
        p.error("give --scene FILE or --all")

    if args.modality == "xray":
        if args.recrop or args.preview:
            p.error("--recrop and --preview are optical-only; not "
                    "supported yet for --modality xray")
        return _main_xray(args, scenes)

    # Would an actual build run on the CPU?  Checked per scene below, AFTER the
    # already-current short-circuit, so that on a GPU-less box a run with
    # nothing to do still succeeds instead of refusing a no-op.  --device cpu
    # counts: there is exactly one way to say "yes, I mean it on CPU", and it is
    # --allow-cpu.  This is the only escape hatch anywhere -- the live server
    # never offers one, because a wedged daemon thread with no cancel endpoint
    # is a far worse place to discover you meant something else.
    refuse_cpu = (not args.allow_cpu) and (
        args.device == "cpu" or (args.device is None and not cuda_available()))

    tile = None if args.tile_size == "auto" else int(args.tile_size)
    opts = dict(axis=args.axis, step_deg=args.step, supersample=args.supersample,
                pan_mm=args.pan_mm, n_cond=args.n_cond, quality=args.quality,
                format=args.format, psf=args.psf == "on")
    root = args.root
    if args.preview:
        # One definition of what a preview is, shared with the server, so a
        # library built here is byte-for-byte one the server accepts as current.
        opts.update(PREVIEW_BUILD)
        if root == DEFAULT_ROOT:
            root = DEFAULT_PREVIEW_ROOT

    rc = 0
    if args.recrop:
        # Deliberately independent of --force and the staleness check: cropping
        # does not change a build key, so a cropped library stays current and a
        # stale one stays stale.
        for s in scenes:
            lib = library_dir(s, root)
            try:
                recrop_library(lib)
            except (FileNotFoundError, ValueError) as exc:
                print(f"[frame-library] FAILED {s}: {exc}", file=sys.stderr)
                rc = 1
        return rc

    for s in scenes:
        lib = library_dir(s, root)
        if not args.force and is_current(s, lib, **build_params(**opts)):
            print(f"[frame-library] {s}: already current -> {lib}")
            continue
        if refuse_cpu:
            print(f"[frame-library] {s}: {CPU_BUILD_REFUSAL}", file=sys.stderr)
            rc = 2
            continue
        print(f"[frame-library] building {s} -> {lib}")
        try:
            man = build_library(s, root=root, tile_size=tile,
                                vram_fraction=args.vram_fraction,
                                device=args.device, **opts)
        except RuntimeError as exc:
            print(f"[frame-library] FAILED {s}: {exc}", file=sys.stderr)
            rc = 1
            continue
        n = len(man["frames"])
        size = sum(os.path.getsize(os.path.join(lib, f["file"]))
                   for f in man["frames"]) / 2**20
        rnd = man["rendered"]
        zmin, zmax = zoom_limits(man)
        print(f"[frame-library] {s}: {n} frames at {rnd['width']}x{rnd['height']}, "
              f"{size:.1f} MB, zoom range {zmin:.2f}..{zmax:.0f}x")
    return rc


if __name__ == "__main__":
    sys.exit(main())
