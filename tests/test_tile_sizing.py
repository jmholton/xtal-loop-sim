"""Trace-tile sizing: the default must be safe on a mesh scene without probing.

A droplet-bearing scene (e.g. a crystal_harvester Hampton loop) is
unrenderable at a flat default tile; the OOM math is in
`et._MESH_BYTES_PER_RAY_FACE`. See docs/DECISIONS.md 2026-08-11 (the mesh
path never culled) for how the budget moved to `_mesh_survivor_chunk`.

The budget is calculated, not measured: `plan_tile_size`'s probing ramp
resets torch's global peak-memory counters that other tools read, and its
upper rungs are exactly what WSL2 silently spills on instead of failing.
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
from loop_sim.renderer.engine_torch import (TSurfaceMesh,
                                            fit_tile_size)

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
# TSurfaceMesh AABB-culls and chunks its own survivors, so `tile_rays x
# faces x 160 B` no longer depends on the caller's tile. The three tests
# below pin that law on `_mesh_survivor_chunk` instead of `fit_tile_size`
# (see docs/DECISIONS.md 2026-08-11). The end-to-end no-OOM test at the
# bottom is what actually proves the pair works.
# ---------------------------------------------------------------------------

@cuda_only
def test_tile_is_no_longer_shrunk_by_face_count():
    """A mesh scene gets the same flat default a tube scene does; pinned
    so the change cannot regress by accident.
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
    """A fraction of free VRAM is not a bound on a card someone else is
    using: sizing purely by fraction spilled a real build into the WSL2
    zone on an idle 16 GB card.  The cull makes survivors scarce enough
    that an absolute cap costs nothing.
    """
    dev = torch.device("cuda")
    # Face counts must straddle where the floor stops fitting the budget
    # (~6,553 faces at a 2 GiB cap); omitting 50,976 (the Rayleigh-matched
    # droplet) would let `max(_MESH_CHUNK_MIN, ...)` silently override the
    # cap again. 200,000 is absurd on purpose.
    for faces in (234, 2880, 5472, 6553, 22464, 50976, 200_000):
        chunk = et._mesh_survivor_chunk(faces, dev, vram_fraction=1.0)
        peak = chunk * faces * et._MESH_BYTES_PER_RAY_FACE
        assert chunk >= 1, f"{faces} faces -> chunk {chunk}"
        assert peak <= et._MESH_CHUNK_MAX_BYTES, f"{faces} faces -> {peak/2**30:.2f} GiB"


@cuda_only
def test_survivor_chunk_floor_yields_to_the_budget():
    """An absurd mesh must still produce a usable chunk, but never one the
    budget cannot pay for. The floor is a preference against a
    pathologically small chunk; past ~6,553 faces it no longer fits and
    must yield.
    """
    dev = torch.device("cuda")
    assert et._mesh_survivor_chunk(500, dev) >= et._MESH_CHUNK_MIN   # fits: honour it
    for faces in (50_976, 200_000):
        chunk = et._mesh_survivor_chunk(faces, dev)
        assert chunk >= 1, f"{faces} faces -> chunk {chunk}; must never be zero"
        assert chunk < et._MESH_CHUNK_MIN, "floor must yield when it cannot fit"
        assert chunk * faces * et._MESH_BYTES_PER_RAY_FACE <= et._MESH_CHUNK_MAX_BYTES

    # Beyond ~13.4M faces (2 GiB / 160 B) even one ray's Moller-Trumbore
    # temporaries exceed the budget, and no chunking can help -- that
    # would need a per-face broad phase, deliberately not built (dead by
    # Amdahl since the AABB cull landed; see docs/DECISIONS.md 2026-08-11).
    # The contract here is only that it stays renderable, never zero.
    assert et._mesh_survivor_chunk(200_000_000, dev) == 1


@cuda_only
def test_the_cull_is_byte_exact_against_brute_force():
    """The whole justification for the AABB cull, asserted directly.

    Every triangle point lies inside the vertex AABB, so a ray the slab
    test rejects provably misses every face -- brute force returned INF
    for exactly those rays. Widening the box to infinity disables the
    cull without touching any other code path, so this compares the two
    answers on identical input.
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
    """End to end: render a droplet-bearing scene with no tile_size
    argument, the call every real caller makes (camera_server,
    frame_library's scout sweep, bench_frame, the investigation
    harnesses). Must not raise torch.OutOfMemoryError.
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


# ---------------------------------------------------------------------------
# The VRAM guarantee targets the beamline's 12 GB TITAN V, not the 16 GB
# dev box these tests run on: "it worked on my box" is not a guarantee.
# torch.cuda.set_per_process_memory_fraction imposes a synthetic ceiling
# so a 12 GB card can be asserted here on whatever hardware CI has.
# ---------------------------------------------------------------------------

import contextlib
import re


@contextlib.contextmanager
def _simulated_card(gb):
    """Constrain this process to `gb` of VRAM, then restore."""
    total = torch.cuda.get_device_properties(0).total_memory
    if gb * 2**30 > total:
        pytest.skip(f"cannot simulate {gb} GB on a {total/2**30:.1f} GB card")
    torch.cuda.empty_cache()
    torch.cuda.set_per_process_memory_fraction(gb * 2**30 / total, 0)
    try:
        yield
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, 0)
        torch.cuda.empty_cache()


@cuda_only
def test_tile_size_shrinks_when_the_card_is_smaller():
    """The tile must come from the card, not from a constant: a version
    that never called `mem_get_info()` would be fine on 16 GB and
    silently wrong on 12.
    """
    dev = torch.device("cuda")
    ts = _FakeScene([], dev)
    big = fit_tile_size(ts, 50_000_000)
    # set_per_process_memory_fraction cannot be used here: it constrains the
    # allocator while mem_get_info keeps reporting the real device, so the
    # sizing would not see it. The budget override is the seam.
    os.environ["LOOPSIM_VRAM_BUDGET_GB"] = "2"
    try:
        small = fit_tile_size(ts, 50_000_000)
    finally:
        del os.environ["LOOPSIM_VRAM_BUDGET_GB"]
    assert small < big, "tile ignored a smaller card"
    assert small >= et._TILE_FIT_MIN


@cuda_only
def test_memory_budget_tracks_free_not_total():
    """A shared 8-GPU node means another tenant's allocation must reduce ours."""
    b = et.memory_budget()
    free, total = torch.cuda.mem_get_info()
    assert b <= free, "budget exceeded what is actually free"
    assert b <= total - et._VRAM_HEADROOM_BYTES + 1


@cuda_only
def test_preflight_accepts_a_build_that_fits_and_refuses_one_that_does_not():
    """The end-to-end guarantee, on a simulated small card.

    A team member re-rendering a scene on the beamline's 12 GB TITAN V
    must get either a build or an actionable refusal, never an OOM
    mid-render and never a WSL2-style silent spill.

    The refusal below is provoked by shrinking the budget, not by growing
    the render: a test for a memory guard must never itself be the
    allocation that breaks the machine, and it does not need to be, since
    the guard compares a render against a budget and either side can move.
    """
    from loop_sim.scene.scene import load
    from loop_sim.renderer.engine_torch import (TorchScene, check_render_fits,
                                                RenderTooLargeError)
    scene_path = os.path.join(REPO_ROOT, "data", "scene_files", "hampton_300um.yaml")
    ts = None
    try:
        sc = load(scene_path, device="cpu")
        sc.camera_cfg = dict(sc.camera_cfg, width=1396, height=644)
        ts = TorchScene(sc, torch.device("cuda"), torch.float64)

        os.environ["LOOPSIM_VRAM_BUDGET_GB"] = "12"      # a TITAN V
        tile = check_render_fits(ts, n_cond=1, psf=False, supersample=1)
        assert tile and tile > 0, "a routine template render must be allowed"

        os.environ["LOOPSIM_VRAM_BUDGET_GB"] = "1.05"    # smaller than the render
        with pytest.raises(RenderTooLargeError) as exc:
            check_render_fits(ts, n_cond=1, psf=False, supersample=4)
        msg = str(exc.value)
        assert "supersample" in msg, "refusal must name the knob to turn"
        m = re.search(r"--supersample (\d+) is the largest", msg)
        assert m and int(m.group(1)) < 4, "suggestion must be below the request"
    finally:
        del ts
        et.release_vram_ceiling()
        os.environ.pop("LOOPSIM_VRAM_BUDGET_GB", None)
        torch.cuda.empty_cache()
