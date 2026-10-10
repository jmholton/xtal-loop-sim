#!/usr/bin/env python
"""Sprinkle flecks of frost onto the nylon of a scene: small faceted ice grains
stuck to the fiber surface, partly buried in it.

Each fleck is an irregular convex polyhedron (an intersection of half-spaces
with random face normals, like add_crystal's lattice solids) so it goes through
the renderer's interval logic exactly as the crystal does.  Flecks are listed
FIRST in the object list so the part that sits inside the fiber (or the drop)
is ice, not nylon.

Usage: /programs/pytorch/envs/pt/bin/python add_ice.py SCENE_IN NAME [--n 30]
       [--size 0.003 0.012] [--seed 1]
writes scene_files/NAME.yaml
"""
import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, "/home/jamesh/projects/loop_sim/claude")
from loop_sim.scene.tube import neville_sample  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("name")
ap.add_argument("--n", type=int, default=30, help="number of flecks on the nylon")
ap.add_argument("--n-drop", type=int, default=0, help="number of flecks on the drop faces")
ap.add_argument("--size", nargs=2, type=float, default=[0.003, 0.012], metavar=("MIN", "MAX"),
                help="fleck diameter range, mm")
ap.add_argument("--faces", type=int, default=12, help="facets per fleck")
ap.add_argument("--seed", type=int, default=1)
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
scene = yaml.safe_load(open(args.scene))
scene["materials"]["ice"] = {"n": 1.31, "mu_optical": 0.5, "mu_xray": 0.03,
                             "color": [0.98, 0.99, 1.0]}

# Fiber centrelines, weighted by arc length so flecks land uniformly on the nylon.
curves = []
for o in scene["objects"]:
    if o["material"] == "nylon" and o["shape"]["type"] == "tube":
        s = o["shape"]
        pts = neville_sample(s["path"], 400)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        curves.append((o["name"], pts, seg, s["diameter"] / 2))
# Only the part of the fiber inside the camera's field (x is unchanged by rotx).
cam, mot = scene["camera"], scene.get("motor", {})
half_w = 0.5 * cam["width"] * cam["pixel_size"]
x_lo, x_hi = -mot.get("tx", 0.0) - half_w, -mot.get("tx", 0.0) + half_w
curves = [(nm, pts, np.where((pts[:-1, 0] > x_lo) & (pts[:-1, 0] < x_hi), seg, 0.0), r)
          for nm, pts, seg, r in curves]
lengths = np.array([c[2].sum() for c in curves])

def fleck(centre, radial, size):
    """Irregular convex ice grain: an intersection of half-spaces around `centre`,
    slightly flattened along `radial` (the surface normal) like a frost platelet.
    Jittered, randomly rotated Fibonacci directions spread over the whole sphere,
    so the solid is always bounded (purely random normals can leave a side open)."""
    j = np.arange(args.faces) + 0.5
    zz = 1 - 2 * j / args.faces
    az = np.pi * (1 + 5 ** 0.5) * j
    n = np.c_[np.sqrt(1 - zz ** 2) * np.cos(az), np.sqrt(1 - zz ** 2) * np.sin(az), zz]
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    n = n @ q.T + rng.normal(scale=0.2, size=n.shape)
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    h = 0.5 * size * rng.uniform(0.75, 1.0, args.faces)
    h *= 1 - 0.35 * np.abs(n @ radial)
    return {"type": "intersection",
            "children": [{"type": "half_space",
                          "normal": [round(float(x), 8) for x in nk],
                          "offset": round(float(nk @ centre + hk), 9)}
                         for nk, hk in zip(n, h)]}


def log_size():
    return np.exp(rng.uniform(*np.log(args.size)))      # log-uniform: mostly small


flecks = []


def add(where, centre, radial, size):
    k = len(flecks)
    flecks.append({"name": f"ice_{k:03d}", "material": "ice", "shape": fleck(centre, radial, size)})
    print(f"ice_{k:03d} on {where:7s} size {size*1e3:4.1f} um  at "
          f"({centre[0]*1e3:7.1f}, {centre[1]*1e3:6.1f}, {centre[2]*1e3:5.1f}) um")


for _ in range(args.n):
    name, pts, seg, r_fib = curves[rng.choice(len(curves), p=lengths / lengths.sum())]
    i = rng.choice(len(seg), p=seg / seg.sum())
    p = pts[i] + rng.random() * (pts[i + 1] - pts[i])
    tan = (pts[i + 1] - pts[i]) / seg[i]
    u = np.cross(tan, [0.0, 0.0, 1.0]) if abs(tan[2]) < 0.9 else np.cross(tan, [1.0, 0.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(tan, u)
    phi = rng.uniform(0, 2 * np.pi)
    radial = np.cos(phi) * u + np.sin(phi) * v
    size = log_size()
    # Centre sits just outside the fiber surface; ~1/4 of the grain is buried.
    add(name, p + radial * (r_fib + 0.25 * size), radial, size)

# Drop faces: area-weighted points on the mesh, both faces (+z and -z), kept off
# the pinned rim where the film is thinner than a grain.
drops = [o for o in scene["objects"] if o["shape"]["type"] == "surface_mesh"]
if args.n_drop and drops:
    V = np.array(drops[0]["shape"]["vertices"], float)
    F = np.array(drops[0]["shape"]["faces"], int)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    fn = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(fn, axis=1)
    fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-30)   # degenerate faces: area 0, never drawn
    fc = (a + b + c) / 3
    fn *= np.sign(np.einsum("ij,ij->i", fn, fc - V.mean(0)))[:, None]   # outward
    zmax = np.abs(V[:, 2]).max()
    w = area * (np.abs(fc[:, 2]) > 0.15 * zmax)
    for _ in range(args.n_drop):
        f = rng.choice(len(F), p=w / w.sum())
        r1, r2 = rng.random(2)
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        p = a[f] + r1 * (b[f] - a[f]) + r2 * (c[f] - a[f])
        size = log_size()
        add("drop", p + fn[f] * 0.25 * size, fn[f], size)

scene["objects"] = flecks + scene["objects"]
out = os.path.join("/home/jamesh/projects/loop_sim/jacob_run", "scene_files", args.name + ".yaml")
with open(out, "w") as fh:
    yaml.safe_dump(scene, fh, default_flow_style=None, sort_keys=False)
print(f"wrote {out}")
