"""
Trace-tile sizing: the DEFAULT must be safe on a mesh scene without probing.

The old default was a flat 1,000,000 rays, so a 640x480 frame went through in a
single pass. On a scene with a solvent droplet that is 307200 x 2880 faces x
160 B = 19.8 GB and a hard OOM -- i.e. every droplet-bearing scene, including a
routine crystal_harvester Hampton loop, was unrenderable at default settings.

The budget is calculated rather than measured on purpose: plan_tile_size's
probing ramp resets torch's global peak-memory counters (which bench_frame.py
and acceptance_voltron.py read) and its upper rungs are exactly the allocations
WSL2 spills on instead of failing, so it cannot be what runs by default.

WHERE THE BUDGET LIVES CHANGED ON 2026-08-11. It used to divide the whole
frame's ray count down until `tile_rays x faces x 160 B` fitted. TSurfaceMesh
now AABB-culls (rejecting ~99.7% of rays on the shipped droplet scene before a
triangle is touched) and chunks its own survivors, so the product no longer
scales with the caller's tile. The tile went back to the flat default and the
budget moved to `_mesh_survivor_chunk`. Same law, same constant, enforced one
level down -- worth 10.2x on the build frame on top of the cull's own 4.05x.

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


# ---------------------------------------------------------------------------
# The budget moved (2026-08-11), the invariant did not.
#
# TSurfaceMesh now AABB-culls and chunks its own survivors, so `tile_rays x
# faces x 160 B` no longer depends on the caller's tile and shrinking the tile
# buys only passes. The three tests below used to pin that law on
# fit_tile_size; they now pin the SAME law on _mesh_survivor_chunk, which is
# where it is enforced. The end-to-end no-OOM test at the bottom is unchanged
# and is what actually proves the pair works.
# ---------------------------------------------------------------------------

@cuda_only
def test_tile_is_no_longer_shrunk_by_face_count():
    """The behaviour change, pinned so it cannot regress by accident.

    A mesh scene gets the same flat default a tube scene does. Measured on
    hampton_300um_realistic at build resolution: the old face-derived tile was
    6800 rays / 133 passes / 19.90 s; one full-frame tile is 1.95 s at 2.56 GB.
    """
    dev = torch.device("cuda")
    WH = 640 * 480
    assert fit_tile_size(_FakeScene([_mesh(500, dev)], dev), WH) == WH
    assert fit_tile_size(_FakeScene([_mesh(5472, dev)], dev), WH) == WH


@cuda_only
def test_survivor_chunk_shrinks_in_proportion_to_face_count():
    dev = torch.device("cuda")
    small = et._mesh_survivor_chunk(500, dev)
    big = et._mesh_survivor_chunk(2000, dev)
    # 4x the faces -> ~1/4 the survivors per chunk (both clear of the floor).
    assert big < small
    assert small / big == pytest.approx(4.0, rel=0.05)


@cuda_only
def test_survivor_chunk_keeps_predicted_peak_inside_the_budget():
    """The point of the whole exercise: the default must not exceed VRAM.

    Checked against the same law the sizing uses, so this pins the arithmetic
    rather than the constant -- the constant itself is a measurement, recorded
    beside _MESH_BYTES_PER_RAY_FACE.
    """
    dev = torch.device("cuda")
    faces = 2880                                   # a crystal_harvester droplet
    free, _ = torch.cuda.mem_get_info()
    chunk = et._mesh_survivor_chunk(faces, dev, vram_fraction=0.50)
    predicted = chunk * faces * et._MESH_BYTES_PER_RAY_FACE
    assert predicted <= free * 0.50


@cuda_only
def test_survivor_chunk_is_capped_absolutely_not_just_as_a_fraction():
    """A fraction of free VRAM is not a bound on a card someone else is using.

    Sizing purely by fraction took ~8 GB on an idle 16 GB card and pushed a
    build to 13.7 GB in nvidia-smi -- inside the WSL2 spill zone, on a GPU
    shared with a desktop. The cull makes survivors scarce enough that the cap
    costs nothing.
    """
    dev = torch.device("cuda")
    # Face counts MUST straddle the point where the floor stops fitting the
    # budget (~6,553 faces at a 2 GiB cap). The first version of this test used
    # 234/2880/5472 -- all below it -- and passed while a 50,976-face droplet
    # reserved 16.1 GB, because `max(_MESH_CHUNK_MIN, ...)` silently overrode
    # the cap. 50,976 is the Rayleigh-matched droplet; 200,000 is absurd on
    # purpose.
    for faces in (234, 2880, 5472, 6553, 22464, 50976, 200_000):
        chunk = et._mesh_survivor_chunk(faces, dev, vram_fraction=1.0)
        peak = chunk * faces * et._MESH_BYTES_PER_RAY_FACE
        assert chunk >= 1, f"{faces} faces -> chunk {chunk}"
        assert peak <= et._MESH_CHUNK_MAX_BYTES, f"{faces} faces -> {peak/2**30:.2f} GiB"


@cuda_only
def test_survivor_chunk_floor_yields_to_the_budget():
    """An absurd mesh must still produce a usable chunk -- but never one the
    budget cannot pay for.

    This previously asserted the floor WINS (`== _MESH_CHUNK_MIN`). That
    assertion was the bug: applying the floor unconditionally made a
    50,976-face droplet demand 2048 x 50976 x 160 B = 16.7 GB and spill the
    card, under a cap that was supposed to be 2 GiB. The floor is a preference
    against a pathologically small chunk; past ~6,553 faces it no longer fits
    and must yield.
    """
    dev = torch.device("cuda")
    assert et._mesh_survivor_chunk(500, dev) >= et._MESH_CHUNK_MIN   # fits: honour it
    for faces in (50_976, 200_000):
        chunk = et._mesh_survivor_chunk(faces, dev)
        assert chunk >= 1, f"{faces} faces -> chunk {chunk}; must never be zero"
        assert chunk < et._MESH_CHUNK_MIN, "floor must yield when it cannot fit"
        assert chunk * faces * et._MESH_BYTES_PER_RAY_FACE <= et._MESH_CHUNK_MAX_BYTES

    # Beyond ~13.4M faces (2 GiB / 160 B) even ONE ray's Moller-Trumbore
    # temporaries exceed the budget, and no chunking can help -- that would need
    # a per-face broad phase, which this codebase deliberately does not have
    # (dead by Amdahl once the AABB cull lands; see DECISIONS). The contract
    # there is only that it stays renderable rather than returning zero.
    assert et._mesh_survivor_chunk(200_000_000, dev) == 1


@cuda_only
def test_the_cull_is_byte_exact_against_brute_force():
    """The whole justification for Step 1, asserted directly.

    Every triangle point lies inside the vertex AABB, so a ray the slab test
    rejects provably misses every face -- brute force returned INF for exactly
    those rays. Widening the box to infinity disables the cull without touching
    any other code path, so this compares the two answers on identical input.
    """
    dev = torch.device("cuda")
    mesh = _mesh(1500, dev)
    rng = np.random.RandomState(7)
    # A mix: some rays aimed through the unit cube the mesh lives in, some not.
    o = torch.as_tensor(np.c_[rng.rand(4000, 2), np.full(4000, -50.0)],
                        device=dev, dtype=torch.float64)
    d = torch.as_tensor(np.tile([0.0, 0.0, 1.0], (4000, 1)),
                        device=dev, dtype=torch.float64)
    culled = mesh.ray_intersect(o, d)
    lo, hi = mesh._bbox_lo.clone(), mesh._bbox_hi.clone()
    try:
        mesh._bbox_lo = torch.full_like(lo, -float("inf"))
        mesh._bbox_hi = torch.full_like(hi, float("inf"))
        brute = mesh.ray_intersect(o, d)
    finally:
        mesh._bbox_lo, mesh._bbox_hi = lo, hi
    names = ("t_enter", "t_exit", "n_enter", "n_exit")
    for nm, a, b in zip(names, culled, brute):
        assert torch.equal(a, b), f"{nm} differs between culled and brute force"
    assert torch.isfinite(culled[0]).sum() > 100, "test exercised too few hits"


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
