#!/usr/bin/env python3
"""
Render the digitized fiber from this.yaml, with an auto-generated twisted stem.

Coordinate convention
---------------------
camera_slow = [0, -1, 0]  →  +y is UP in the image (row 0 = top = negative y in lab).
The digitizer already stores waypoints with  dy = -(row − oy) × px_mm,
which gives +dy for points above the origin — consistent with this convention.

Translation formula (motor tx / ty):
    ty = (origin_row − H/2) × px_mm   (negative when origin is above centre)
"""
import sys, os
sys.path.insert(0, '/home/jamesh/projects/loop_sim/claude')

import numpy as np
import yaml
from PIL import Image

from loop_sim.scene.scene          import load as load_scene
from loop_sim.renderer.microscope  import render
from loop_sim.motors.goniometer    import Goniometer
from crystal_harvester.nylon_mechanics import helix_path

# ---------------------------------------------------------------------------
# Image / camera parameters (matching real_loop_himag.jpg)
# ---------------------------------------------------------------------------
W, H  = 704, 480
px_um = 0.8233
px_mm = px_um / 1000.0
fd_mm = 0.020   # fiber diameter (mm)

# ---------------------------------------------------------------------------
# Origin and waypoints  (clean values extracted from this.yaml print output)
# ---------------------------------------------------------------------------
origin_col, origin_row = 445.2, 196.4

# tx: right of image centre  →  +x
# ty: (row - H/2) * px_mm  →  negative when above centre (top of image = −y)
tx =  (origin_col - W / 2) * px_mm
ty =  (origin_row - H / 2) * px_mm   # NOT negated — camera_slow=[0,-1,0] makes +y=up

loop_path = [
    [ 0.0000,  0.0000, 0.0],
    [-0.1634,  0.0415, 0.0],
    [-0.2912,  0.0282, 0.0],
    [-0.3183, -0.0277, 0.0],
    [-0.2832, -0.0793, 0.0],
    [-0.1789, -0.0836, 0.0],
    [ 0.0000,  0.0000, 0.0],   # closed loop
]

# ---------------------------------------------------------------------------
# Stem direction: opposite to centroid of waypoints (axis of symmetry)
# ---------------------------------------------------------------------------
wp = np.array(loop_path[1:-1], dtype=float)   # skip origin and closing point
centroid  = wp.mean(axis=0)
stem_axis = -centroid / np.linalg.norm(centroid)   # points away from loop

stem_length   = 0.7          # mm
pitch         = 20 * fd_mm   # loosely wound (pitch_ratio = 20)
helix_radius  = fd_mm / 2    # touching fibres: 2R = fd_mm (center-to-center = diameter)

stem1 = helix_path(helix_radius, pitch, stem_length, n_points=40,
                   phase_offset=0.0, axis=stem_axis)
stem2 = helix_path(helix_radius, pitch, stem_length, n_points=40,
                   phase_offset=np.pi, axis=stem_axis)

# ---------------------------------------------------------------------------
# Read user-editable scene settings (camera, geometry, materials) from
# thisyaml_scene.yaml, then inject the computed stem paths.
# thisyaml_scene.yaml is NEVER overwritten — edit it freely.
# ---------------------------------------------------------------------------
scene_yaml = os.path.join(os.path.dirname(__file__), 'thisyaml_scene.yaml')
with open(scene_yaml) as f:
    scene_dict = yaml.safe_load(f)

geometry = scene_dict.get("geometry", {})

# Replace objects: loop from hardcoded path + auto-generated stems
scene_dict["objects"] = [
    {
        "name":     "loop_fiber",
        "material": "nylon",
        "shape": {"type": "tube", "diameter": fd_mm, "path": loop_path, "n_samples": 100},
    },
    {
        "name":     "stem_fiber_1",
        "material": "nylon",
        "shape": {"type": "tube", "diameter": fd_mm,
                  "path": stem1.tolist(), "n_samples": 80},
    },
    {
        "name":     "stem_fiber_2",
        "material": "nylon",
        "shape": {"type": "tube", "diameter": fd_mm,
                  "path": stem2.tolist(), "n_samples": 80},
    },
]

# Write to a temp file — thisyaml_scene.yaml is untouched
tmp_yaml = '/tmp/thisyaml_render_scene.yaml'
with open(tmp_yaml, 'w') as f:
    yaml.dump(scene_dict, f, default_flow_style=None, sort_keys=False)

# ---------------------------------------------------------------------------
# Load and render
# ---------------------------------------------------------------------------
scene = load_scene(tmp_yaml)
gonio = Goniometer(geometry)
gonio.set(tx=tx, ty=ty)

print(f"Stem axis: ({stem_axis[0]:+.4f}, {stem_axis[1]:+.4f}, {stem_axis[2]:+.4f})", file=sys.stderr)
print(f"Origin offset: tx={tx:+.5f} mm, ty={ty:+.5f} mm", file=sys.stderr)
print(f"Rendering {W}×{H} ...", file=sys.stderr)

img_arr, jpeg_bytes = render(scene, gonio, n_cond=1)

out_path = os.path.join(os.path.dirname(__file__), 'thisyaml_render.jpg')
Image.fromarray((img_arr * 255).astype(np.uint8)).save(out_path)
print(f"Saved → {out_path}", file=sys.stderr)
