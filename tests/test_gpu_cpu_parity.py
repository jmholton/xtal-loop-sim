"""
GPU/CPU parity tests for the float64 intersection fix.

Regression guard for the "hairy/spikey fiber" bug: the CUDA Tube/SurfaceMesh
intersection used float32, which suffered catastrophic cancellation in the
intersection quadratic (amplified by the 50 mm ray-march origin offset),
producing off-surface hits -> wrong normals -> NA/TIR binary flips along the
fiber silhouette. The fix computes those quadratics in float64 on the GPU.

These tests require CUDA; they skip cleanly on a GPU-less box (where the numpy
float64 CPU path remains the ground-truth reference).

Run:  pytest tests/test_gpu_cpu_parity.py -v
"""
import os
import sys

import numpy as np
import pytest

# Make `loop_sim` importable when pytest is run from anywhere.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


cuda_only = pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")

HAMPTON = os.path.join(REPO_ROOT, "scene_files", "hampton_300um.yaml")


def _render(device, roty=0.0, n_cond=1):
    """Render hampton at a goniometer pose; return uint8-scaled (H, W, 3) int array."""
    from loop_sim.scene.scene import load
    from loop_sim.motors.goniometer import Goniometer
    from loop_sim.renderer.microscope import render

    scene = load(HAMPTON, device=device)
    gonio = Goniometer(scene.geometry).set(roty=roty)
    img, _ = render(scene, gonio, n_cond=n_cond)
    return (img * 255).clip(0, 255).astype(int)


@cuda_only
@pytest.mark.parametrize("roty", [0.0, 45.0])
def test_tube_gpu_matches_cpu(roty):
    """
    Fixed GPU tube path must match the numpy float64 CPU reference.

    Pre-fix this scene/pose diverged on ~200 silhouette pixels (full 0<->255
    flips). Post-fix only a handful of exact-threshold grazing pixels may remain
    (torch-CUDA-f64 vs numpy-f64 reduction-order differences), so we allow a
    small budget that is still ~10x below the divergence and ~1000x below the
    documented float32 edge tolerance.
    """
    cpu = _render("cpu", roty=roty)
    gpu = _render("cuda", roty=roty)
    dmax = np.abs(cpu - gpu).max(axis=2)
    n_big = int((dmax > 10).sum())
    assert n_big <= 30, (
        f"roty={roty}: {n_big} pixels diverge >10 (speckle regression?); "
        f"max diff={int(np.abs(cpu - gpu).max())}"
    )


@cuda_only
def test_mesh_cuda_float64_matches_numpy():
    """
    SurfaceMesh CUDA path (now float64) must match the numpy float64 path on a
    loop-scale cube viewed from 50 mm away (the cancellation regime). Also guards
    the line-198 dtype fix: a float32 torch.zeros() default would raise here.
    """
    from loop_sim.scene.surface_mesh import SurfaceMesh

    s = 0.05
    V = s * np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], dtype=float)
    F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                  [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
                  [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7]], dtype=int)

    xs = np.linspace(-0.08, 0.08, 90)
    gx, gy = np.meshgrid(xs, xs)
    o = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, -50.0)], axis=1)
    d = np.tile(np.array([0.0, 0.0, 1.0]), (len(o), 1))

    cpu = SurfaceMesh(V, F, device="cpu")
    gpu = SurfaceMesh(V, F, device="cuda")  # builds float64 tensors

    c = cpu._mt_batch(o, d)          # numpy float64
    g = gpu._mt_batch(o, d)          # torch float64 on CUDA (would crash pre-dtype-fix)
    names = ["t_min", "t_max", "fi_min", "fi_max", "t_back", "fi_bwd"]

    assert np.isfinite(np.asarray(c[0], float)).sum() > 0, "test fired no real hits"

    # Distances must agree to ~f64 epsilon.
    for nm, ca, ga in zip(names, c, g):
        if not nm.startswith("t"):
            continue
        ca = np.asarray(ca, float)
        ga = np.asarray(ga, float)
        fin = np.isfinite(ca) & np.isfinite(ga)
        assert int((np.isfinite(ca) ^ np.isfinite(ga)).sum()) == 0, f"{nm} hit/miss mismatch"
        assert np.abs(ca[fin] - ga[fin]).max() < 1e-9, f"{nm} t-value divergence"

    # Face-index diffs are allowed ONLY when the looked-up normal is identical
    # (coplanar triangles of the same cube face) -> zero rendering impact.
    fn = cpu._face_normals
    for idx_name in ("fi_min", "fi_max", "fi_bwd"):
        i = names.index(idx_name)
        cfi = np.asarray(c[i], int)
        gfi = np.asarray(g[i], int)
        differ = cfi != gfi
        if differ.any():
            assert np.abs(fn[cfi[differ]] - fn[gfi[differ]]).max() < 1e-12, (
                f"{idx_name} selects a non-coplanar face (normal differs)"
            )
