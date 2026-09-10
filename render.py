#!/usr/bin/env python3
"""
Render a loop-sim scene YAML file.

Usage:
    python3 render.py SCENE.yaml [--tx TX] [--ty TY] [--n-cond N] [--output OUT.jpg]

Motor settings (tx, ty) are read from the YAML 'motor:' section if present,
and can be overridden on the command line.
"""
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    p.add_argument('--tx',   type=float, default=None, help='X translation (mm), overrides YAML')
    p.add_argument('--ty',   type=float, default=None, help='Y translation (mm), overrides YAML')
    p.add_argument('--rotx', type=float, default=None, help='Rotation about x-axis (degrees)')
    p.add_argument('--roty', type=float, default=None, help='Rotation about y-axis (degrees)')
    p.add_argument('--rotz', type=float, default=None, help='Rotation about z-axis (degrees)')
    p.add_argument('--n-cond', type=int, default=1,    help='Condenser rays (default 1)')
    p.add_argument('--device', default='cpu',
                   help='Compute device: cpu (default) or cuda. cuda uses the '
                        'GPU-resident engine_torch engine (the same one '
                        'camera_server uses), byte-identical to the CPU '
                        'reference in float64.')
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

    # 640, matching microscope.py / engine_torch.py / beam.py / camera_server.py
    # and every shipped scene: the BL831 pixels are 1.11 non-square, so
    # 640 x 7.4 um and 704 x 6.7324 um cover the same field to 0.08%, while
    # 704 x 7.4 um would over-cover it by +9.9%.  See loop_sim/renderer/field.py
    # SENSOR_WH.
    W = camera.get('width',  640)
    H = camera.get('height', 480)

    tx   = args.tx   if args.tx   is not None else float(motor.get('tx',   0.0))
    ty   = args.ty   if args.ty   is not None else float(motor.get('ty',   0.0))
    rotx = args.rotx if args.rotx is not None else float(motor.get('rotx', 0.0))
    roty = args.roty if args.roty is not None else float(motor.get('roty', 0.0))
    rotz = args.rotz if args.rotz is not None else float(motor.get('rotz', 0.0))

    # engine_torch mirrors a CPU-loaded scene onto the device itself (see
    # TorchScene / build_torch_shape) -- always load 'cpu' here and let it do
    # that, matching camera_server / loop_sim.library / the test suite.
    scene = load_scene(args.scene, device='cpu' if args.device == 'cuda' else args.device)
    gonio = Goniometer(geometry)
    gonio.set(tx=tx, ty=ty, rotx=rotx, roty=roty, rotz=rotz)

    print(f"Scene:  {args.scene}", file=sys.stderr)
    print(f"Motor:  tx={tx:+.5f} mm  ty={ty:+.5f} mm  "
          f"rotx={rotx:.1f}°  roty={roty:.1f}°  rotz={rotz:.1f}°",
          file=sys.stderr)
    print(f"Rendering {W}×{H}, n_cond={args.n_cond} ...", file=sys.stderr)

    if args.device == 'cuda':
        # The resident GPU engine (loop_sim/renderer/engine_torch.py) -- the
        # same one camera_server uses -- rather than the legacy per-object
        # CUDA path in scene/tube.py + scene/surface_mesh.py. Falls back to
        # CPU transparently if no CUDA device is visible, same as
        # camera_server's own device selection.
        import torch
        from loop_sim.renderer.engine_torch import TorchScene, render_torch
        dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"Device: {dev} (engine_torch)", file=sys.stderr)
        tscene = TorchScene(scene, dev, torch.float64)
        img_arr = render_torch(tscene, gonio, n_cond=args.n_cond).detach().cpu().numpy()
    else:
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
