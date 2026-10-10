#!/usr/bin/env python
"""Rebuild James's hi-mag scene for Jacob's camera server with the BL831 pixel aspect.

hoop.yaml was digitized from real_loop_himag.jpg at 0.8233 um/px on both axes, but
the hi-mag sample camera's pixels are 0.8233 (h) x 0.9139 (v) um.  This scales the
digitized y by 0.9139/0.8233 so the geometry is physical, reruns the original
pipeline (add_stem / add_droplet / add_crystal, same arguments as scene.yaml), and
renders on a square-pixel 634x480 x 0.9139 um camera, which the server resamples
onto the 704x480 sensor raster (field.to_sensor) -- Jacob's convention for his
640 x 7.4 um mid-mag scenes.  Field of view: 579.4 x 438.7 um vs the real
579.6 x 438.7 um.

Usage: /programs/pytorch/envs/pt/bin/python make_scene.py [--name N] [crystal args]
  default: the plate from scene.yaml (--preset plate --dim 0.04 0.01)
  rod:     --name james_himag_rod --preset needle --dim 0.010 0.050 --c-tilt 15 10
  thick:   ... rod args ... --name james_himag_rod_thick --volume 0.00233
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import yaml

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="james_himag")
ap.add_argument("--preset", default="plate")
ap.add_argument("--dim", nargs="+", default=["0.04", "0.01"])
ap.add_argument("--c-tilt", nargs=2, type=float, metavar=("IN_PLANE", "OUT_OF_PLANE"),
                help="orient the crystal c axis: degrees from the spindle (+x) toward "
                     "+y in the image plane, then toward +z out of it")
ap.add_argument("--volume", default="0.001",
                help="droplet volume in mm^3 (0.001 -> 37.4 um/side; 0.00233 -> 74.8, i.e. 2x)")
ap.add_argument("--drop-half-thickness", type=float, default=None, metavar="MM",
                help="reshape the drop to the small-slope Young-Laplace surface pinned to "
                     "the same rim, with this peak thickness per side (overrides --volume)")
ap.add_argument("--nylon-mu", type=float, default=None, metavar="PER_MM",
                help="nylon mu_optical (scene.yaml has 0.1; 20 fits real_loop_himag.jpg)")
args = ap.parse_args()


def laplace_reshape(verts, peak, grid_mm=0.001):
    """Replace each vertex's |z| with peak * u(x, y) / max(u), where del^2 u = -1 inside
    the rim and u = 0 on it.  add_droplet's radially-stretched cap keeps full height
    where the teardrop narrows (a 1 um-wide, 30 um-tall fin at the crossover); a
    pinned constant-curvature film thins with the local width instead."""
    from matplotlib.path import Path
    from scipy.interpolate import RegularGridInterpolator
    from scipy.sparse import diags, identity, kron
    from scipy.sparse.linalg import spsolve

    v = np.asarray(verts, dtype=float)
    rim = v[np.abs(v[:, 2]) < 1e-12, :2]
    c = rim.mean(0)
    rim = rim[np.argsort(np.arctan2(rim[:, 1] - c[1], rim[:, 0] - c[0]))]
    x = np.arange(rim[:, 0].min() - 2 * grid_mm, rim[:, 0].max() + 3 * grid_mm, grid_mm)
    y = np.arange(rim[:, 1].min() - 2 * grid_mm, rim[:, 1].max() + 3 * grid_mm, grid_mm)
    X, Y = np.meshgrid(x, y, indexing="ij")
    inside = Path(rim).contains_points(np.c_[X.ravel(), Y.ravel()])
    nx, ny = X.shape

    def lap1d(n):
        return diags([np.ones(n - 1), -2 * np.ones(n), np.ones(n - 1)], [-1, 0, 1])
    L = kron(lap1d(nx), identity(ny)) + kron(identity(nx), lap1d(ny))
    idx = np.flatnonzero(inside)
    A = (L.tocsr()[idx][:, idx]).tocsc()        # Dirichlet u = 0 outside the rim
    u = np.zeros(nx * ny)
    u[idx] = spsolve(-A, np.full(len(idx), grid_mm ** 2))
    U = RegularGridInterpolator((x, y), u.reshape(nx, ny), bounds_error=False,
                                fill_value=0.0)
    uv = U(v[:, :2])
    out = v.copy()
    out[:, 2] = np.sign(v[:, 2]) * peak * uv / u.max()
    return out


def mesh_volume(verts, faces):
    v = np.asarray(verts, dtype=float)
    f = np.asarray(faces)
    return abs(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum()) / 6.0

SRC = "/home/jamesh/projects/loop_sim/claude"
OUT = "/home/jamesh/projects/loop_sim/jacob_run"     # work dir (scripts live in the repo)
PY = sys.executable

PX_H_UM, PX_V_UM = 0.8233, 0.9139
Y_SCALE = PX_V_UM / PX_H_UM
SENSOR_W, SENSOR_H = 704, 480
CAM_W = round(SENSOR_W * PX_H_UM / PX_V_UM)          # 634

work = os.path.join(OUT, "components", args.name)
os.makedirs(work, exist_ok=True)

hoop = yaml.safe_load(open(os.path.join(SRC, "hoop.yaml")))
hoop["fiber"]["waypoints"] = [[x, round(y * Y_SCALE, 6), z]
                              for x, y, z in hoop["fiber"]["waypoints"]]
hoop_path = os.path.join(work, "hoop_physical.yaml")
with open(hoop_path, "w") as f:
    yaml.safe_dump(hoop, f, default_flow_style=None, sort_keys=False)


def run(script, *args):
    subprocess.run([PY, os.path.join(SRC, script), hoop_path, *args], check=True,
                   stdout=subprocess.DEVNULL)


run("add_stem.py", "--output", os.path.join(work, "loop.yaml"))
run("add_droplet.py", "--volume", args.volume, "--output", os.path.join(work, "droplet.yaml"))
axes = []
if args.c_tilt:
    th, ph = np.radians(args.c_tilt)
    c = np.array([np.cos(ph) * np.cos(th), np.cos(ph) * np.sin(th), np.sin(ph)])
    a = np.cross([0.0, 0.0, 1.0], c)
    a /= np.linalg.norm(a)
    b = np.cross(c, a)
    for flag, v, length in (("--a-axis", a, 10.5), ("--b-axis", b, 10.5),
                            ("--c-axis", c, 27.3)):
        axes += [flag, *[f"{x:.6f}" for x in v * length]]
run("add_crystal.py", "--preset", args.preset, "--dim", *args.dim, *axes,
    "--output", os.path.join(work, "crystal.yaml"))

loop = yaml.safe_load(open(os.path.join(work, "loop.yaml")))
crystal = yaml.safe_load(open(os.path.join(work, "crystal.yaml")))
droplet = yaml.safe_load(open(os.path.join(work, "droplet.yaml")))
if args.drop_half_thickness:
    dshape = droplet["objects"][0]["shape"]
    new_v = laplace_reshape(dshape["vertices"], args.drop_half_thickness)
    dshape["vertices"] = np.round(new_v, 9).tolist()
    print(f"drop reshaped (Young-Laplace, small slope): peak {args.drop_half_thickness*1e3:.1f} "
          f"um/side, volume {mesh_volume(new_v, dshape['faces'])*1e3:.2f} nL")

scene = yaml.safe_load(open(os.path.join(SRC, "scene.yaml")))
if args.nylon_mu is not None:
    scene["materials"]["nylon"]["mu_optical"] = args.nylon_mu
scene["camera"].update(width=CAM_W, height=SENSOR_H, pixel_size=PX_V_UM * 1e-3)
# Photo pose: the crossover sat at sensor pixel (445.2, 196.4).
oc, orow = hoop["origin_px"]["col"], hoop["origin_px"]["row"]
scene["motor"] = {"tx": round((oc - SENSOR_W / 2) * PX_H_UM * 1e-3, 6),
                  "ty": round(-(SENSOR_H / 2 - orow) * PX_V_UM * 1e-3, 6)}
# Priority order: crystal before droplet.
scene["objects"] = loop["objects"] + crystal["objects"] + droplet["objects"]

out = os.path.join(OUT, "scene_files", args.name + ".yaml")
with open(out, "w") as f:
    yaml.safe_dump(scene, f, default_flow_style=None, sort_keys=False)
print(f"wrote {out}")
print(f"camera {CAM_W}x{SENSOR_H} @ {PX_V_UM} um  y_scale {Y_SCALE:.5f}  "
      f"motor {scene['motor']}")
print("objects:", [o["name"] for o in scene["objects"]])
