"""CLI: build (or refresh) a pre-computed frame library.

    python -m loop_sim.library --scene scene_files/hampton_300um.yaml
    python -m loop_sim.library --all            # every scene in scene_files/
    python -m loop_sim.library --scene X --force

Libraries land in frame_library/<scene_stem>/ and are tracked in git -- they
are part of the deliverable, not build output.
"""
import argparse
import glob
import os
import sys

from .frame_library import (DEFAULT_MARGIN, DEFAULT_N_COND, DEFAULT_QUALITY,
                            DEFAULT_ROOT, DEFAULT_STEP_DEG, build_library,
                            is_current, library_dir)


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
    p.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                   help="render this much larger than the camera so panning "
                        f"is a crop (default {DEFAULT_MARGIN})")
    p.add_argument("--n-cond", type=int, default=DEFAULT_N_COND)
    p.add_argument("--quality", type=int, default=DEFAULT_QUALITY)
    p.add_argument("--force", action="store_true", help="rebuild even if current")
    args = p.parse_args(argv)

    scenes = list(args.scene)
    if args.all:
        scenes += sorted(glob.glob(os.path.join("scene_files", "*.yaml")))
    if not scenes:
        p.error("give --scene FILE or --all")

    rc = 0
    for s in scenes:
        lib = library_dir(s, args.root)
        if not args.force and is_current(s, lib):
            print(f"[frame-library] {s}: already current -> {lib}")
            continue
        print(f"[frame-library] building {s} -> {lib}")
        try:
            man = build_library(s, root=args.root, axis=args.axis,
                                step_deg=args.step, margin=args.margin,
                                n_cond=args.n_cond, quality=args.quality)
        except RuntimeError as exc:
            print(f"[frame-library] FAILED {s}: {exc}", file=sys.stderr)
            rc = 1
            continue
        n = len(man["frames"])
        size = sum(os.path.getsize(os.path.join(lib, f["file"]))
                   for f in man["frames"]) / 2**20
        print(f"[frame-library] {s}: {n} frames, {size:.1f} MB, "
              f"pan margin +-{man['pan_px']['x']}x{man['pan_px']['y']} px")
    return rc


if __name__ == "__main__":
    sys.exit(main())
