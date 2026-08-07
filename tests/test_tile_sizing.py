"""
Trace-tile sizing: the DEFAULT must be safe on a mesh scene without probing.

The old default was a flat 1,000,000 rays, so a 640x480 frame went through in a
single pass. On a scene with a solvent droplet that is 307200 x 2880 faces x
160 B = 19.8 GB and a hard OOM -- i.e. every droplet-bearing scene, including a
routine crystal_harvester Hampton loop, was unrenderable at default settings.

The default now computes the tile from the largest mesh and free VRAM.
Calculated rather than measured on purpose: plan_tile_size's probing ramp
resets torch's global peak-memory counters (which bench_frame.py and
acceptance_voltron.py read) and its upper rungs are exactly the allocations
WSL2 spills on instead of failing, so it cannot be what runs by default.

Run:  pytest tests/test_tile_sizing.py -v
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

torch = pytest.importorskip("torch")

from loop_sim.renderer import engine_torch as et
from loop_sim.renderer.engine_torch import (TIntersection, TSurfaceMesh,
                                            _mesh_face_count, fit_tile_size)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                               reason="CUDA not available")


def _mesh(n_faces, dev):
    """A throwaway mesh with exactly n_faces triangles."""
    v = np.random.RandomState(0).rand(n_faces * 3, 3)
    f = np.arange(n_faces * 3).reshape(n_faces, 3)
    return TSurfaceMesh(v, f, dev, torch.float64)


class _FakeScene:
    """Just enough TorchScene for the sizing helpers: a device and shapes."""
    def __init__(self, shapes, dev):
        self.shapes, self.dev, self.dt = shapes, dev, torch.float64


def test_mesh_face_count_finds_the_largest_mesh():
    dev = torch.device("cpu")
    shapes = [_mesh(10, dev), _mesh(70, dev), _mesh(30, dev)]
    assert _mesh_face_count(shapes) == 70, "must take the max, not the sum"


def test_mesh_face_count_walks_csg_children():
    """A droplet reaches the renderer wrapped in CSG nodes, not bare.

    Missing it there would report 0 faces and hand back the flat default --
    exactly the OOM this sizing exists to prevent, and silently.
    """
    dev = torch.device("cpu")
    nested = TIntersection([TIntersection([_mesh(2880, dev)])])
    assert _mesh_face_count([nested]) == 2880


def test_mesh_face_count_is_zero_without_a_mesh():
    assert _mesh_face_count([]) == 0


@cuda_only
def test_meshless_scene_keeps_the_flat_default():
    """Tube and primitive scenes must be untouched by this change.

    Every fps and parity number in the repo is measured on hampton_300um, a
    tube scene: if its tile changed, a frame would be split into passes it was
    not split into before and the benchmarks would silently shift.
    """
    ts = _FakeScene([], torch.device("cuda"))
    WH = 640 * 480
    assert fit_tile_size(ts, WH) == WH                    # one pass, as before
    assert fit_tile_size(ts, 4_000_000) == et._TILE_DEFAULT


@cuda_only
def test_tile_shrinks_in_proportion_to_face_count():
    dev = torch.device("cuda")
    small = fit_tile_size(_FakeScene([_mesh(500, dev)], dev), 10_000_000)
    big = fit_tile_size(_FakeScene([_mesh(2000, dev)], dev), 10_000_000)
    # 4x the faces -> ~1/4 the rays per tile (both well clear of the floor).
    assert big < small
    assert small / big == pytest.approx(4.0, rel=0.05)


@cuda_only
def test_tile_keeps_predicted_peak_inside_the_budget():
    """The point of the whole exercise: the default must not exceed VRAM.

    Checked against the same law the sizing uses, so this pins the arithmetic
    rather than the constant -- the constant itself is a measurement, recorded
    beside _MESH_BYTES_PER_RAY_FACE.
    """
    dev = torch.device("cuda")
    faces = 2880                                   # a crystal_harvester droplet
    ts = _FakeScene([_mesh(faces, dev)], dev)
    WH = 640 * 480
    free, _ = torch.cuda.mem_get_info()
    tile = fit_tile_size(ts, WH, vram_fraction=0.80)
    assert tile < WH, "a 2880-face scene must not go through in one pass"
    predicted = tile * faces * et._MESH_BYTES_PER_RAY_FACE
    assert predicted <= free * 0.80


@cuda_only
def test_tile_never_falls_below_the_floor():
    """An absurd mesh must still produce a usable tile, not zero."""
    dev = torch.device("cuda")
    ts = _FakeScene([_mesh(200_000, dev)], dev)
    assert fit_tile_size(ts, 640 * 480) == et._TILE_FIT_MIN


@cuda_only
def test_default_render_of_a_mesh_scene_does_not_oom():
    """End to end: render a droplet-bearing scene with NO tile_size argument.

    This is the call every caller in the repo makes -- camera_server,
    frame_library's scout sweep, bench_frame, the investigation harnesses -- and
    before this change it raised torch.OutOfMemoryError on any scene with a
    droplet.
    """
    from loop_sim.motors.goniometer import Goniometer
    from loop_sim.renderer.engine_torch import TorchScene, render_torch
    from loop_sim.scene.materials import AIR, Material
    from loop_sim.scene.scene import Scene, SceneObject
    from loop_sim.scene.surface_mesh import SurfaceMesh

    # A sphere-ish mesh dense enough to have OOM'd at the old flat default:
    # 2880 faces x 307200 rays x 160 B = 19.8 GB.
    u, v = np.mgrid[0:np.pi:37j, 0:2*np.pi:41j]
    pts = np.stack([0.15*np.sin(u)*np.cos(v), 0.15*np.sin(u)*np.sin(v),
                    0.075*np.cos(u)], -1).reshape(-1, 3)
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts)
    mesh = SurfaceMesh(hull.points, hull.simplices)
    assert len(hull.simplices) > 2000, f"only {len(hull.simplices)} faces"

    geom = {"camera_fast": [1, 0, 0], "camera_slow": [0, 1, 0],
            "beam_axis": [0, 0, 1], "optical_axis": [0, 0, -1],
            "rotx_axis": [1, 0, 0], "roty_axis": [0, 1, 0], "rotz_axis": [0, 0, 1]}
    cam = {"width": 640, "height": 480, "pixel_size": 0.0074,
           "na_objective": 0.10, "na_condenser": 0.07}
    water = Material(name="solvent", n=1.33, mu_optical=0.0, mu_xray=0.0,
                     color=(1.0, 1.0, 1.0))
    scene = Scene([SceneObject(name="drop", shape=mesh, material=water)],
                  geom, cam, {}, background=AIR)
    ts = TorchScene(scene, torch.device("cuda"), torch.float64)
    try:
        img = render_torch(ts, Goniometer(geom).set(), n_cond=1)   # no tile_size
        assert img.shape == (480, 640, 3)
        assert np.isfinite(img.cpu().numpy()).all()
    finally:
        del ts
        torch.cuda.empty_cache()
