#!/usr/bin/env python
"""Beam-weighted volume of crystal, nylon and drop vs horizontal beam position.

The number of photons a material scatters is

    N = Phi * t * (scattering per um of path in that material) * P,
    P(x) = sum_i  f_i(x) * L_i * dA                       [um^3 per um^2 = um]

where Phi is the beam flux, f is the beam profile normalised to unit integral
(the fraction of incident photons per um^2), and L_i is ray i's path length
through the material.  P is the material's volume weighted by the beam, per
incident photon: the mean thickness of it that one beam photon passes through.
At fixed flux, P for the crystal is proportional to the Bragg photons.  P for
the nylon and the drop is proportional to the background they scatter (diffuse,
not Bragg); the per-um scattering constant differs between materials and is
not applied here.

Path lengths come from Jacob's scene tracer (Scene.path_segments, straight rays,
since X-rays do not refract).  Object priority is honoured, so the drop's volume
excludes the crystal.  The tracer runs once per spindle angle on a grid of beam
rays in the image plane, centred on the crystal; each Gaussian beam is then a
weighted sum over that grid.  X-ray absorption is ignored: about 1-2 % through
tens of um of protein or water at typical energies.  The scan is horizontal
(along camera_fast, the spindle axis), with x = 0 and the beam's vertical centre
on the crystal centre.

Run with Jacob's venv:
  cd /home/jadoughty/projects/loop_sim_MINE/xtal-loop-sim && env PYTHONPATH=. \\
      .venv/bin/python /home/jamesh/projects/loop_sim/jacob_run/beam_scan.py SCENE \\
      [--fwhm H V [H V ...]] [--aperture W H [W H ...]] [--phi 0 80] [--range 150]
      [--step 0.5] [--grid 1]
writes probe/<stem>_beam_scan.csv and probe/<stem>_beam_scan.png in the work dir
"""
import argparse
import os
import time
from itertools import combinations

import numpy as np
import yaml
from scipy.spatial import ConvexHull

from loop_sim.scene.scene import load

ap = argparse.ArgumentParser()
ap.add_argument("scene")
ap.add_argument("--fwhm", nargs="+", type=float, default=[14.0, 105.0, 100.0, 50.0],
                help="beam FWHM pairs, um: H V [H V ...]")
ap.add_argument("--aperture", nargs="+", type=float, default=None,
                help="rectangular collimator per beam, um: W H [W H ...]; 0 0 = none.  "
                     "Sharp-edged, centred on the beam and moving with it")
ap.add_argument("--phi", nargs="+", type=float, default=[0.0], help="spindle angles, deg")
ap.add_argument("--range", type=float, default=150.0, help="scan +/- this far, um")
ap.add_argument("--step", type=float, default=0.5, help="scan step, um")
ap.add_argument("--grid", type=float, default=1.0, help="ray spacing, um")
args = ap.parse_args()
assert len(args.fwhm) % 2 == 0, "--fwhm takes H V pairs"
apert = args.aperture or [0.0, 0.0] * (len(args.fwhm) // 2)
assert len(apert) == len(args.fwhm), "--aperture takes one W H pair per beam"
BEAMS = [tuple(args.fwhm[i:i + 2]) + tuple(apert[i:i + 2]) for i in range(0, len(args.fwhm), 2)]
K = 2 * np.sqrt(2 * np.log(2))
MATS = ["crystal", "nylon", "solvent"]
LABEL = {"crystal": "crystal (Bragg)", "nylon": "nylon loop", "solvent": "drop (solvent)"}

WORK = "/home/jamesh/projects/loop_sim/jacob_run"
stem = os.path.splitext(os.path.basename(args.scene))[0]
spec = yaml.safe_load(open(args.scene))
g = spec["geometry"]
fast, slow, beam = (np.array(g[k], float) for k in ("camera_fast", "camera_slow", "beam_axis"))
scene = load(args.scene)

# Crystal centre and exact volume from its half-spaces (um), as a check on the trace.
cr = next(o for o in spec["objects"] if o["material"] == "crystal")
N = np.array([c["normal"] for c in cr["shape"]["children"]], float)
d = np.array([c["offset"] for c in cr["shape"]["children"]], float) * 1e3
V = [np.linalg.solve(N[[i, j, k]], d[[i, j, k]]) for i, j, k in combinations(range(len(N)), 3)
     if abs(np.linalg.det(N[[i, j, k]])) > 1e-9]
V = np.array([p for p in V if np.all(N @ p - d <= 1e-6)])
centre = V.mean(0)
vol_true = ConvexHull(V).volume
print(f"crystal: volume {vol_true:,.0f} um^3, centre {np.round(centre, 1)} um")

xs = np.arange(-args.range, args.range + args.step / 2, args.step)
reach_u = args.range + 1.7 * max(b[0] for b in BEAMS)       # Gaussian tails to < 0.1 %
reach_v = 1.7 * max(b[1] for b in BEAMS)
gu = np.arange(-reach_u, reach_u + args.grid / 2, args.grid)
gv = np.arange(-reach_v, reach_v + args.grid / 2, args.grid)
GU, GV = np.meshgrid(gu, gv)                                # (nv, nu)



def window(g, centre, width):
    """Fraction of each grid cell inside a sharp aperture [centre +/- width/2]."""
    if width <= 0:
        return np.ones_like(g)
    lo, hi = centre - width / 2, centre + width / 2
    h = args.grid / 2
    return np.clip(np.minimum(g + h, hi) - np.maximum(g - h, lo), 0, None) / args.grid


def transmission(bh, bv, aw, ah):
    """Fraction of the Gaussian beam's photons that pass the collimator."""
    from math import erf, sqrt
    t = 1.0
    for f, a in ((bh, aw), (bv, ah)):
        if a > 0:
            t *= erf(a / 2 / (sqrt(2) * f / K))
    return t


rows = []
for phi in args.phi:
    c, s = np.cos(np.radians(phi)), np.sin(np.radians(phi))
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])        # rotx: sample -> lab
    lab = (centre @ R.T + GU.ravel()[:, None] * fast + GV.ravel()[:, None] * slow
           - 2000.0 * beam) * 1e-3                          # mm, 2 mm upstream
    t0 = time.time()
    segs = scene.path_segments(lab @ R, np.tile(R.T @ beam, (len(lab), 1)), t_max=5.0)
    Lmap = {m: np.zeros(len(lab)) for m in MATS}
    for i, sg in enumerate(segs):
        for mat, length in sg:
            if mat.name in Lmap:
                Lmap[mat.name][i] += length * 1e3           # um
    Lmap = {m: L.reshape(GU.shape) for m, L in Lmap.items()}
    dA = args.grid ** 2
    print(f"phi {phi:g}: {len(lab):,} rays traced in {time.time() - t0:.0f} s; volumes in "
          f"window " + ", ".join(f"{m} {L.sum() * dA:,.0f}" for m, L in Lmap.items())
          + f" um^3 (crystal exact {vol_true:,.0f})")
    for bh, bv, aw, ah in BEAMS:
        sx, sy = bh / K, bv / K
        # Profile normalised to unit integral AFTER the collimator: equal flux on the
        # sample, so beams with and without a collimator compare directly.
        fy = np.exp(-gv ** 2 / (2 * sy ** 2)) * window(gv, 0.0, ah)
        fy /= fy.sum() * args.grid                                # 1/um
        for m in MATS:
            col = fy @ Lmap[m] * args.grid                  # (nu,) um per um of u
            for x in xs:
                fx = np.exp(-(gu - x) ** 2 / (2 * sx ** 2)) * window(gu, x, aw)
                fx /= fx.sum() * args.grid
                rows.append((phi, bh, bv, aw, ah, MATS.index(m), x, float(fx @ col * args.grid)))

rows = np.array(rows)
out = os.path.join(WORK, "probe", f"{stem}_beam_scan")
np.savetxt(out + ".csv", rows, delimiter=",", fmt=["%g", "%g", "%g", "%g", "%g", "%d", "%g", "%.5f"],
           header="phi_deg,fwhm_h_um,fwhm_v_um,aperture_w_um,aperture_h_um,"
                  "material(0=crystal 1=nylon 2=solvent),"
                  "beam_x_um,beam_weighted_volume_per_photon_um  (sum f*L*dA, f = unit-integral "
                  "Gaussian; proportional to scattered photons at fixed flux)", comments="# ")


def curve(phi, b, m):
    r = rows[(rows[:, 0] == phi) & np.all(rows[:, 1:5] == b, axis=1) & (rows[:, 5] == MATS.index(m))]
    return r[:, 6], r[:, 7]


def beam_label(b):
    bh, bv, aw, ah = b
    lab = f"{bh:g} × {bv:g} µm FWHM beam"
    if aw > 0 or ah > 0:
        lab += f", {aw:g} × {ah:g} µm collimator ({100 * transmission(*b):.0f}% passes)"
    return lab


for phi in args.phi:
    for b in BEAMS:
        x, y = curve(phi, b, "crystal")
        k = np.argmax(y)
        msg = ", ".join(f"{m} {np.interp(x[k], *curve(phi, b, m)):.2f}" for m in MATS)
        print(f"phi {phi:g}, {beam_label(b)}: at crystal peak (x = {x[k]:g} um) {msg} um")

# ---- plot: one panel per beam, one fixed colour per material -----------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = dict(zip(MATS, ["#2a78d6", "#eb6834", "#1baf7a"]))   # reference slots 1-3, validated
phi = args.phi[0]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, len(BEAMS), figsize=(max(5.2 * len(BEAMS), 7.6), 5.0), dpi=150,
                         sharey=True, squeeze=False)
fig.patch.set_facecolor(SURF)
ymax = max(curve(phi, b, m)[1].max() for b in BEAMS for m in MATS)
for ax, b in zip(axes[0], BEAMS):
    ax.set_facecolor(SURF)
    for m in MATS:
        x, y = curve(phi, b, m)
        ax.plot(x, y, color=COLOR[m], lw=2, solid_capstyle="round")
        k = np.argmax(y)
        ax.plot(x[k], y[k], "o", ms=7, color=COLOR[m], mec=SURF, mew=2, zorder=3)
        ax.annotate(f"{y[k]:.2f}", (x[k], y[k]), xytext=(0, 8), textcoords="offset points",
                    ha="center", color=INK2, fontsize=9.5)
    ax.set_title(beam_label(b).replace(", ", ",\n", 1) + "  (H × V)", color=INK, fontsize=10.5,
                 loc="left")
    ax.set_xlim(xs[0], xs[-1])
    # Log: the drop is ~20x the crystal; a linear axis flattens the crystal curve.
    ax.set_yscale("log")
    ax.set_ylim(0.05, ymax * 2.5)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("beam position relative to crystal centre (µm)", color=INK2)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, length=0)
axes[0][0].set_ylabel("beam-weighted volume per photon (µm³/µm²)", color=INK2)
fig.suptitle("Beam-weighted volume vs horizontal beam position\n"
             f"∝ photons scattered at equal flux;  {stem},  φ = {phi:g}°,  log scale",
             color=INK, fontsize=12, x=0.01, y=0.98, ha="left", va="top")
fig.legend([plt.Line2D([], [], color=COLOR[m], lw=2) for m in MATS], [LABEL[m] for m in MATS],
           frameon=False, loc="upper left", ncol=3, labelcolor=INK2, bbox_to_anchor=(0.0, 0.875))
fig.tight_layout(rect=(0, 0, 1, 0.8))
fig.savefig(out + ".png", facecolor=SURF)
print(f"wrote {out}.csv and {out}.png")
