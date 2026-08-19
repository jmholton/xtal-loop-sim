"""
X-ray attenuation (Beer-Lambert) tests for the beam reporter + radiograph.

Covers the Phase-1-of-enhancements work that finally consumes the long-dead
`mu_xray` material coefficient:

  * Scene.path_segments()        — ordered front-to-back traversal (the new
                                   primitive path_lengths is refactored on top of)
  * beam.compute_beam_volumes()  — per-material absorbed_dose + transmitted_frac
                                   + top-level beam_transmission, shadowing-aware
  * X-ray transmission map       — render_xray_numpy (CPU ref) and, when torch is
                                   present, render_xray_torch parity vs the numpy ref

These use tiny synthetic scenes (boxes/spheres) with analytically known path
lengths, so the asserts are exact Beer-Lambert, not golden images.

Run:  pytest tests/test_beam_attenuation.py -v
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loop_sim.scene.materials import Material, AIR
from loop_sim.scene.primitives import HalfSpace, Sphere
from loop_sim.scene.csg import Intersection
from loop_sim.scene.scene import Scene, SceneObject
from loop_sim.motors.goniometer import Goniometer
from loop_sim.renderer.beam import compute_beam_volumes, render_xray_numpy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GEOM = {
    "camera_fast": [1, 0, 0],
    "camera_slow": [0, 1, 0],
    "beam_axis":   [0, 0, 1],
    "optical_axis": [0, 0, -1],
    "rotx_axis":   [1, 0, 0],
    "roty_axis":   [0, 1, 0],
    "rotz_axis":   [0, 0, 1],
}
# Flat profile → all mini-beams equally weighted; small grid for speed.
BEAM_CFG = {"profile": "flat", "spacing": 0.01, "fwhm_x": 0.05, "fwhm_y": 0.03}


def _mat(name, mu_xray):
    return Material(name=name, n=1.5, mu_optical=0.0, mu_xray=mu_xray,
                    color=(1.0, 1.0, 1.0))


def _slab(z0, thickness, mat, name):
    """Infinite slab perpendicular to the beam (z), spanning z0..z0+thickness.

    Built from two half-spaces so the axial beam (dir = +z) hits non-degenerately
    (Box's AABB slab method NaNs for a ray exactly parallel to a face axis).
    Infinite in x/y, so every mini-beam crosses the full `thickness`.
    """
    z1 = z0 + thickness
    shape = Intersection(HalfSpace(normal=(0, 0, -1), offset=-z0),   # z >= z0
                         HalfSpace(normal=(0, 0,  1), offset=z1))     # z <= z1
    return SceneObject(name, shape, mat)


def _scene(objects, camera_cfg=None):
    return Scene(objects, GEOM, camera_cfg or {}, BEAM_CFG, background=AIR)


def _gonio():
    return Goniometer(GEOM)   # at rest → identity transform


def _sum_absorbed(res):
    return sum(v["absorbed_dose"] for k, v in res.items()
              if k != "beam_transmission")


# ---------------------------------------------------------------------------
# path_segments — ordering + path_lengths equivalence
# ---------------------------------------------------------------------------

def test_path_segments_front_to_back_order():
    a = SceneObject("A", Sphere(centre=(0, 0, -2.0), radius=0.5), _mat("A", 1.0))
    b = SceneObject("B", Sphere(centre=(0, 0,  2.0), radius=0.5), _mat("B", 1.0))
    scene = _scene([a, b])

    o = np.array([[0.0, 0.0, -50.0]])
    d = np.array([[0.0, 0.0,   1.0]])
    segs = scene.path_segments(o, d)[0]

    names = [m.name for (m, _L) in segs]
    # A (closer, hit first) must precede B; air gaps interleave.
    non_air = [n for n in names if n != "air"]
    assert non_air == ["A", "B"]
    # chord through each sphere centre = 2r = 1.0
    lens = {m.name: L for (m, L) in segs if m.name != "air"}
    assert lens["A"] == pytest.approx(1.0, abs=1e-6)
    assert lens["B"] == pytest.approx(1.0, abs=1e-6)


def test_path_lengths_equals_reduced_path_segments():
    a = SceneObject("A", Sphere(centre=(0, 0, -2.0), radius=0.5), _mat("A", 1.0))
    b = SceneObject("B", Sphere(centre=(0, 0,  2.0), radius=0.5), _mat("B", 1.0))
    scene = _scene([a, b])
    o = np.array([[0.0, 0.0, -50.0]])
    d = np.array([[0.0, 0.0,   1.0]])

    pl = scene.path_lengths(o, d)[0]
    reduced = {}
    for m, L in scene.path_segments(o, d)[0]:
        reduced[m] = reduced.get(m, 0.0) + L
    assert pl == reduced


# ---------------------------------------------------------------------------
# beam.compute_beam_volumes — Beer-Lambert attenuation
# ---------------------------------------------------------------------------

def test_single_slab_transmission_matches_analytic():
    mu, L = 2.1, 0.5
    scene = _scene([_slab(-L / 2, L, _mat("crystal", mu), "crystal")])
    res = compute_beam_volumes(scene, _gonio())

    assert res["beam_transmission"] == pytest.approx(np.exp(-mu * L), rel=1e-6)
    assert res["crystal"]["transmitted_frac"] == pytest.approx(np.exp(-mu * L), rel=1e-6)
    assert res["crystal"]["absorbed_dose"] == pytest.approx(1.0 - np.exp(-mu * L), rel=1e-6)
    # Energy book-keeping: absorbed + transmitted = 1
    assert _sum_absorbed(res) + res["beam_transmission"] == pytest.approx(1.0, abs=1e-9)


def test_air_slab_is_transparent():
    scene = _scene([_slab(-0.25, 0.5, _mat("air2", 0.0), "air2")])
    res = compute_beam_volumes(scene, _gonio())
    assert res["beam_transmission"] == pytest.approx(1.0, abs=1e-9)
    assert res["air2"]["absorbed_dose"] == pytest.approx(0.0, abs=1e-12)
    assert res["air2"]["transmitted_frac"] == pytest.approx(1.0, abs=1e-9)


def test_stacked_slabs_downstream_is_shadowed():
    muA, LA = 1.0, 0.3
    muB, LB = 2.0, 0.4
    front = _slab(-2.0, LA, _mat("front", muA), "front")
    back  = _slab( 1.0, LB, _mat("back",  muB), "back")
    res = compute_beam_volumes(_scene([front, back]), _gonio())

    tA = np.exp(-muA * LA)
    tB = np.exp(-muB * LB)
    # Front slab: standard absorption from full incident flux.
    assert res["front"]["absorbed_dose"] == pytest.approx(1.0 - tA, rel=1e-6)
    # Back slab sees flux already attenuated by the front slab.
    assert res["back"]["absorbed_dose"] == pytest.approx(tA * (1.0 - tB), rel=1e-6)
    assert res["beam_transmission"] == pytest.approx(tA * tB, rel=1e-6)
    assert _sum_absorbed(res) + res["beam_transmission"] == pytest.approx(1.0, abs=1e-9)

    # Shadowing: the same back slab alone absorbs MORE (it sees full flux).
    res_b_only = compute_beam_volumes(_scene([back]), _gonio())
    assert res_b_only["back"]["absorbed_dose"] > res["back"]["absorbed_dose"]
    assert res_b_only["back"]["absorbed_dose"] == pytest.approx(1.0 - tB, rel=1e-6)


# ---------------------------------------------------------------------------
# X-ray transmission map (radiograph)
# ---------------------------------------------------------------------------

def _xray_scene():
    cam = {"width": 32, "height": 24, "pixel_size": 0.05}
    sphere = SceneObject("blob", Sphere(centre=(0, 0, 0), radius=0.5),
                         _mat("blob", 5.0))
    return _scene([sphere], camera_cfg=cam)


def test_xray_numpy_map_casts_a_shadow():
    scene = _xray_scene()
    T = render_xray_numpy(scene, _gonio())
    assert T.shape == (24, 32)
    assert np.all(T >= 0.0) and np.all(T <= 1.0 + 1e-9)
    centre = T[12, 16]            # ray through the sphere centre
    corner = T[0, 0]             # ray that misses the sphere
    # centre chord = 2r = 1.0 → exp(-5*1.0)
    assert centre == pytest.approx(np.exp(-5.0), rel=1e-3)
    assert corner == pytest.approx(1.0, abs=1e-9)
    assert centre < corner


def test_xray_torch_matches_numpy():
    torch = pytest.importorskip("torch")
    from loop_sim.renderer.engine_torch import TorchScene
    from loop_sim.renderer.xray_torch import render_xray_torch

    scene = _xray_scene()
    gonio = _gonio()
    T_np = render_xray_numpy(scene, gonio)

    dev = torch.device("cpu")
    ts = TorchScene(scene, dev, torch.float64)
    T_t = render_xray_torch(ts, gonio).cpu().numpy()

    assert np.allclose(T_np, T_t, atol=1e-9)
