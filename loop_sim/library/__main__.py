"""CLI: build (or refresh) a pre-computed frame library.

    python -m loop_sim.library --scene scene_files/hampton_300um.yaml
    python -m loop_sim.library --all            # every scene in scene_files/
    python -m loop_sim.library --scene X --force

Libraries land in frame_library/<scene_stem>/ and are tracked in git -- they
are part of the deliverable, not build output.

Tile size defaults to `auto`, which calibrates peak memory against two cheap
probe renders and sizes each pass to fit inside --vram-fraction of free VRAM.
"""
import argparse
import glob
import os
import sys

from .frame_library import (DEFAULT_FORMAT, DEFAULT_N_COND, DEFAULT_PAN_MM,
                            DEFAULT_QUALITY, DEFAULT_ROOT, DEFAULT_STEP_DEG,
                            DEFAULT_SUPERSAMPLE, DEFAULT_VRAM_FRACTION,
                            build_library, build_params, is_current,
                            library_dir, zoom_limits)


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
    args = p.parse_args(argv)

    scenes = list(args.scene)
    if args.all:
        scenes += sorted(glob.glob(os.path.join("scene_files", "*.yaml")))
    if not scenes:
        p.error("give --scene FILE or --all")

    tile = None if args.tile_size == "auto" else int(args.tile_size)
    opts = dict(axis=args.axis, step_deg=args.step, supersample=args.supersample,
                pan_mm=args.pan_mm, n_cond=args.n_cond, quality=args.quality,
                format=args.format, psf=args.psf == "on")

    rc = 0
    for s in scenes:
        lib = library_dir(s, args.root)
        if not args.force and is_current(s, lib, **build_params(**opts)):
            print(f"[frame-library] {s}: already current -> {lib}")
            continue
        print(f"[frame-library] building {s} -> {lib}")
        try:
            man = build_library(s, root=args.root, tile_size=tile,
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
