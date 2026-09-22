"""
Rung-2 differential parity for the GPU-resident torch engine (Phase 2).

torch-cpu-float64 must equal the numpy-float64 reference to f64 reduction-order
tolerance. This isolates *port* correctness from *precision* effects and needs
no GPU, so it runs in CI on any box. (Rung-3 torch-cuda vs numpy is a separate
GPU test added later.)

Covers: analytic primitives (2a) and CSG + composite Cylinder + the real
hampton `pin` CSG tree (2b).
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

torch = pytest.importorskip("torch")

from loop_sim.scene import primitives as P
from loop_sim.scene import csg as C
from loop_sim.scene.scene import load
from loop_sim.renderer.engine_torch import build_torch_shape

DEV, DT = torch.device("cpu"), torch.float64
_RNG = np.random.default_rng(0)
HAMPTON = os.path.join(REPO_ROOT, "data", "scene_files", "hampton_300um.yaml")


def _uniform_rays(n=40000, lo=-2.0, hi=2.0):
    o = _RNG.uniform(lo, hi, (n, 3))
    d = _RNG.normal(0, 1, (n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return o, d


def _aimed_rays(centre, n=40000, R=5.0, jit=0.4):
    """Rays fired at `centre` from a surrounding shell, jittered -> mix of hit/miss."""
    centre = np.asarray(centre, float)
    u = _RNG.normal(0, 1, (n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    o = centre + R * u
    d = (centre - o) + _RNG.normal(0, jit, (n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return o, d


def _assert_parity(shape, o, d, rtol=1e-7, ntol=1e-6):
    te, tx, ne, nx = shape.ray_intersect(o, d)
    ts = build_torch_shape(shape, DEV, DT)
    gte, gtx, gne, gnx = (x.numpy() for x in
                          ts.ray_intersect(torch.from_numpy(o), torch.from_numpy(d)))
    label = type(shape).__name__
    for a, b, nm in [(te, gte, "t_enter"), (tx, gtx, "t_exit")]:
        # inf/-inf ends must match exactly; finite values to relative tol
        # (f64 reduction-order noise is amplified on grazing rays with huge |t|).
        same_inf = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
        close = np.isclose(a, b, rtol=rtol, atol=1e-8)
        bad = int((~(close | same_inf)).sum())
        assert bad == 0, f"{label} {nm}: {bad} value/pattern mismatches"
    he, hx = np.isfinite(te), np.isfinite(tx)
    assert int(he.sum()) > 100, f"{label}: test exercised too few hits"
    assert np.abs(ne[he] - gne[he]).max() < ntol, f"{label} entry-normal divergence"
    assert np.abs(nx[hx] - gnx[hx]).max() < ntol, f"{label} exit-normal divergence"


_PRIMS = {
    "sphere": P.Sphere(centre=(0.1, -0.2, 0.3), radius=1.2),
    "halfspace": P.HalfSpace(normal=(0.3, 0.5, -0.8), offset=0.2),
    "infinite_cylinder": P.InfiniteCylinder(centre=(0., 0.1, 0.), axis=(0.2, 1., 0.1), radius=0.7),
    "ellipsoid": P.Ellipsoid(centre=(0., 0., 0.), radii=(1.0, 0.6, 1.4)),
    "box": P.Box(lo=(-0.5, -0.7, -0.3), hi=(0.6, 0.4, 0.9)),
    "capsule": P.Capsule(p0=(-0.5, 0., 0.), p1=(0.5, 0.2, 0.1), radius=0.3),
}

_CSG = {
    "cylinder": (P.Cylinder(centre=(0.2, 0., 0.), axis=(1., .15, 0.), radius=.35, height=2.), (.2, 0, 0)),
    "intersection": (C.Intersection(P.Sphere((0, 0, 0), 1.0), P.Sphere((.4, .1, 0), .9)), (.2, 0, 0)),
    "union": (C.Union(P.Sphere((-.3, 0, 0), .8), P.Sphere((.3, 0, 0), .8)), (0, 0, 0)),
    "difference": (C.Difference(P.Sphere((0, 0, 0), 1.0), P.Sphere((.3, .2, 0), .6)), (0, 0, 0)),
}


@pytest.mark.parametrize("name", list(_PRIMS))
def test_primitive_parity(name):
    _assert_parity(_PRIMS[name], *_uniform_rays())


@pytest.mark.parametrize("name", list(_CSG))
def test_csg_parity(name):
    shape, centre = _CSG[name]
    _assert_parity(shape, *_aimed_rays(centre))


def test_hampton_pin_csg_parity():
    pin = next(ob.shape for ob in load(HAMPTON, device="cpu").objects
               if ob.name == "pin")
    _assert_parity(pin, *_aimed_rays((3.7, 0.0, 0.0)))


# --- 2c: resident Tube + SurfaceMesh kernels ---

def _tube_rays(curve, r, n=30000):
    """Rays from z=-50 mm along +z, targeting points near random curve points."""
    rng = np.random.default_rng(1)
    idx = rng.integers(0, len(curve), n)
    c = curve[idx].copy()
    c[:, 0] += rng.uniform(-4 * r, 4 * r, n)
    c[:, 1] += rng.uniform(-4 * r, 4 * r, n)
    o = c.copy(); o[:, 2] -= 50.0
    d = np.tile([0.0, 0.0, 1.0], (n, 1)) + rng.normal(0, 1e-3, (n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return o, d


@pytest.mark.parametrize("tube_name", ["loop_fiber", "stem_fiber_1", "stem_fiber_2"])
def test_tube_parity(tube_name):
    ob = next(o for o in load(HAMPTON, device="cpu").objects if o.name == tube_name)
    o, d = _tube_rays(np.asarray(ob.shape._curve_pts, float), ob.shape.radius)
    # normals at 50 mm grazing hits carry ~1e-6 reduction-order noise (unit vectors)
    _assert_parity(ob.shape, o, d, rtol=1e-6, ntol=1e-4)


def test_surface_mesh_parity():
    from loop_sim.scene.surface_mesh import SurfaceMesh
    s = 0.05
    V = s * np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                      [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
    Fc = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                   [2, 3, 7], [2, 7, 6], [1, 2, 6], [1, 6, 5], [3, 0, 4], [3, 4, 7]], int)
    mesh = SurfaceMesh(V, Fc, device="cpu")
    _assert_parity(mesh, *_aimed_rays((0.0, 0.0, 0.0)))
