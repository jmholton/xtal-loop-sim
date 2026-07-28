"""Objective scene checks that need no reference photograph.

  A. Dimensional ground truth — a feature of known physical size must span
     size/pixel_size pixels in the rendered image.
  B. NA sensitivity — is an opaque solvent droplet real absorption, or rays
     being discarded by the objective NA gate?

Usage (from the repo root):
    PYTHONPATH=$PWD python investigation/scene_dimcheck.py [mesh_scene.yaml]

The mesh scene is optional; without it only check A runs.  Generate one with
    python -m crystal_harvester.cli --loop-type hampton --loop-size 300 \
        --crystal hexagonal -o /tmp/hampton_300_hex.yaml
"""
import os, sys
import numpy as np, torch
from loop_sim.scene.scene import load
from loop_sim.motors.goniometer import Goniometer
from loop_sim.renderer.engine_torch import TorchScene, render_torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_SCENE = sys.argv[1] if len(sys.argv) > 1 else None
dev = torch.device("cuda")


def shot(path, n_cond=7, scale=1, **motors):
    sc = load(path, device="cpu")
    W0, H0 = sc.camera_cfg["width"], sc.camera_cfg["height"]
    sc.camera_cfg["width"], sc.camera_cfg["height"] = W0 // scale, H0 // scale
    ts = TorchScene(sc, dev, torch.float64)
    g = Goniometer(sc.geometry); g.set(**motors)
    a = render_torch(ts, g, n_cond=n_cond).cpu().numpy()
    del ts; torch.cuda.empty_cache()
    return a, sc


def runs(mask):
    """Return (start, length) of each True run in a 1-D mask."""
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            out.append((i, j - i)); i = j
        else:
            i += 1
    return out


print("=" * 70)
print("A. DIMENSIONAL GROUND TRUTH — bundled hampton_300um.yaml, rotx=0")
print("=" * 70)
a, sc = shot(f"{REPO}/scene_files/hampton_300um.yaml", rotx=0.0)
px_um = sc.camera_cfg["pixel_size"] * 1000
H, W, _ = a.shape
g = a.mean(axis=2)
dark = g < 0.9
ys = np.where(dark.any(axis=1))[0]
xs = np.where(dark.any(axis=0))[0]
print(f"image {W}x{H}, pixel {px_um:.2f} um, frame centre ({W//2},{H//2})")
print(f"all non-white content: cols {xs.min()}..{xs.max()}, rows {ys.min()}..{ys.max()}")

yc = (ys.min() + ys.max()) // 2
print(f"\nhorizontal cut through row {yc} — dark runs (start px, length px, length um):")
for s, l in runs(dark[yc]):
    print(f"    x={s:4d}  len={l:4d} px = {l*px_um:8.1f} um")

# vertical cut through the loop: pick the column with the tallest dark run left of the pin
sub = dark[:, : xs.min() + 120]
col = int(np.argmax(sub.sum(axis=0)))
print(f"\nvertical cut through column {col} — dark runs:")
for s, l in runs(dark[:, col]):
    print(f"    y={s:4d}  len={l:4d} px = {l*px_um:8.1f} um")

print("\nEXPECTED at this pixel size:")
for name, um in [("loop outer dia 300um", 300), ("fiber dia 20um", 20),
                 ("pin dia 700um", 700)]:
    print(f"    {name:24s} -> {um/px_um:7.2f} px")

if MESH_SCENE is None:
    print("\n(no mesh scene given -- skipping the NA-sensitivity check; "
          "pass one as argv[1], see this file's docstring)")
    raise SystemExit(0)

print()
print("=" * 70)
print(f"B. WHY IS THE MESH DROPLET BLACK?  ({os.path.basename(MESH_SCENE)}, 1/8 res)")
print("=" * 70)
print("Sweeping objective NA. Steeply rising brightness => the drop is dark")
print("because rays are culled by the NA gate, not because it absorbs.")
print("(1/8 resolution because droplet meshes exhaust VRAM at full size --")
print(" see docs/HANDOFF.md risk A.)\n")
for na in (0.10, 0.25, 0.50, 0.90):
    sc = load(MESH_SCENE, device="cpu")
    sc.camera_cfg["na_objective"] = na
    sc.camera_cfg["width"] //= 8; sc.camera_cfg["height"] //= 8
    ts = TorchScene(sc, dev, torch.float64)
    gg = Goniometer(sc.geometry); gg.set(rotx=0.0)
    a = render_torch(ts, gg, n_cond=1).cpu().numpy()
    del ts; torch.cuda.empty_cache()
    H, W, _ = a.shape
    core = a[H//3:2*H//3, W//3:2*W//3]
    print(f"  NA_obj={na:4.2f}  drop-core mean={core.mean():.4f}  frame mean={a.mean():.4f}")
