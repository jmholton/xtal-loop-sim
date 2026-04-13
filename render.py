#!/usr/bin/env python3
"""
Render a loop-sim scene YAML file.

Usage:
    python3 render.py SCENE.yaml [--tx TX] [--ty TY] [--n-cond N] [--output OUT.jpg]

Motor settings (tx, ty) are read from the YAML 'motor:' section if present,
and can be overridden on the command line.
"""
import sys, os, argparse
sys.path.insert(0, '/home/jamesh/projects/loop_sim/claude')

import numpy as np
import yaml
from PIL import Image

from loop_sim.scene.scene         import load as load_scene
from loop_sim.renderer.microscope import render
from loop_sim.motors.goniometer   import Goniometer


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('scene', help='Scene YAML file to render')
    p.add_argument('--tx',     type=float, default=None, help='X translation (mm), overrides YAML')
    p.add_argument('--ty',     type=float, default=None, help='Y translation (mm), overrides YAML')
    p.add_argument('--n-cond', type=int,   default=1,    help='Condenser rays (default 1)')
    p.add_argument('--output', default=None,
                   help='Output JPEG path (default: <scene_basename>.jpg in same directory)')
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.scene) as f:
        scene_dict = yaml.safe_load(f)

    geometry = scene_dict.get('geometry', {})
    camera   = scene_dict.get('camera',   {})
    motor    = scene_dict.get('motor',    {})

    W = camera.get('width',  704)
    H = camera.get('height', 480)

    tx = args.tx if args.tx is not None else float(motor.get('tx', 0.0))
    ty = args.ty if args.ty is not None else float(motor.get('ty', 0.0))

    scene = load_scene(args.scene)
    gonio = Goniometer(geometry)
    gonio.set(tx=tx, ty=ty)

    print(f"Scene:  {args.scene}", file=sys.stderr)
    print(f"Motor:  tx={tx:+.5f} mm  ty={ty:+.5f} mm", file=sys.stderr)
    print(f"Rendering {W}×{H}, n_cond={args.n_cond} ...", file=sys.stderr)

    img_arr, jpeg_bytes = render(scene, gonio, n_cond=args.n_cond)

    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(args.scene)[0]
        out_path = base + '.jpg'

    Image.fromarray((img_arr * 255).astype(np.uint8)).save(out_path)
    print(f"Saved  → {out_path}", file=sys.stderr)


if __name__ == '__main__':
    main()
