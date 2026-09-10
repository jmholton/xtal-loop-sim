"""Mesh-back validation of generated scenes: the geometry a scene emits
must match what was requested.

Guards against silently substituting a wrong shape when a solver fails
(the silent-hemisphere regression, docs/DECISIONS.md 2026-08-07): these
tests pin the closed-form droplet replacement (crystal_harvester/
droplet.py) and the validator (crystal_harvester/validate.py) that makes
any such regression loud.

Pure geometry: no rendering, no GPU, no scene files touched.
"""
import copy
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from crystal_harvester.droplet import (biconvex_lens_profile, cap_volume,
                                       droplet_in_loop, mesh_volume,
                                       revolve_biconvex)
from crystal_harvester.hampton_loops import build_hampton_scene
from crystal_harvester.mitegen_mounts import build_mitegen_scene
from crystal_harvester.validate import SceneValidationError, validate_scene

VOLUME = 0.002   # the CLI default --solvent-volume


@pytest.fixture(scope="module")
def scene():
    return build_hampton_scene(solvent_volume_mm3=VOLUME)


@pytest.fixture(scope="module")
def report(scene):
    return validate_scene(scene, requested_volume_mm3=VOLUME)


# ---------------------------------------------------------------------------
# The closed-form droplet itself
# ---------------------------------------------------------------------------

def test_cap_volume_inverts_exactly():
    """The h(V) solve must round-trip through the analytic cap volume."""
    r, z, h, rho = biconvex_lens_profile(0.15, VOLUME, n_z=30)
    assert cap_volume(h, 0.15) == pytest.approx(VOLUME / 2.0, rel=1e-9)
    # cap geometry: the sphere of radius rho must pass through the rim
    assert rho == pytest.approx((0.15**2 + h**2) / (2 * h), rel=1e-12)


def test_biconvex_profile_straddles_and_pins():
    r, z, h, rho = biconvex_lens_profile(0.15, VOLUME, n_z=30)
    assert z[0] == pytest.approx(+h) and z[-1] == pytest.approx(-h)
    assert r[0] == 0.0 and r[-1] == 0.0
    assert r[len(r) // 2] == pytest.approx(0.15)          # rim at z = 0
    assert z[len(z) // 2] == pytest.approx(0.0)


def test_unpinnable_volume_raises_not_falls_back():
    """The old solver silently shipped a hemisphere; impossible input must
    now raise."""
    with pytest.raises(ValueError, match="hemisphere capacity"):
        biconvex_lens_profile(0.15, 0.05, n_z=30)
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    loop = np.stack([0.16 * np.cos(t), 0.16 * np.sin(t), 0 * t], -1)
    with pytest.raises(ValueError):
        droplet_in_loop(loop, 0.020, 0.05)


def test_revolved_mesh_is_watertight_with_no_degenerate_faces():
    r, z, h, rho = biconvex_lens_profile(0.15, VOLUME, n_z=20)
    v, f = revolve_biconvex(r, z, 32)
    # every directed edge exactly once, every undirected edge exactly twice
    directed = {}
    for tri in f:
        for e in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            directed[e] = directed.get(e, 0) + 1
    assert all(n == 1 for n in directed.values())
    und = {}
    for (a, b) in directed:
        k = (min(a, b), max(a, b))
        und[k] = und.get(k, 0) + 1
    assert all(n == 2 for n in und.values())
    areas = 0.5 * np.linalg.norm(
        np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    assert areas.min() > 1e-12


def test_droplet_in_loop_hits_requested_volume():
    """Divergence-theorem volume of the emitted mesh matches the request --
    the check whose absence let a 3.5x-too-big hemisphere ship."""
    t = np.linspace(0, 2 * np.pi, 60, endpoint=False)
    loop = np.stack([0.16 * np.cos(t) - 0.1, 0.16 * np.sin(t), 0 * t], -1)
    v, f, info = droplet_in_loop(loop, 0.020, VOLUME)
    assert mesh_volume(v - np.mean(v, axis=0), f) == pytest.approx(
        VOLUME, rel=1e-6)
    assert info["volume_mm3"] == pytest.approx(VOLUME, rel=1e-6)


# ---------------------------------------------------------------------------
# The generated Hampton scene
# ---------------------------------------------------------------------------

def test_default_scene_validates(report):
    assert report["volume_mm3"] == pytest.approx(VOLUME, rel=0.02)


def test_droplet_sits_in_the_aperture_not_at_the_stem(scene, report):
    """The old generator wrote solver vertices verbatim: drop at the origin
    (the loop/stem junction) while the aperture is at x ~ -0.25 mm."""
    sv = np.array(scene["objects"][1]["shape"]["vertices"])
    lp = np.array(scene["objects"][2]["shape"]["path"])
    assert scene["objects"][1]["name"] == "solvent"
    assert scene["objects"][2]["name"] == "loop_fiber"
    # entirely inside the loop's x-span, nowhere near the stem side
    assert sv[:, 0].max() < 1e-3
    assert sv[:, 0].min() > lp[:, 0].min() - 1e-3
    # rim pinned on the fiber, all the way around
    assert report["rim_to_fiber_mm"]["max"] <= 0.020


def test_droplet_straddles_the_loop_plane(report):
    """A one-sided dome is the hemisphere-fallback signature."""
    assert report["h_above_mm"] > 1e-3
    assert report["h_below_mm"] > 1e-3
    assert report["h_above_mm"] == pytest.approx(report["h_below_mm"],
                                                 rel=1e-3)


def test_crystal_is_centred_in_the_droplet(scene, report):
    cx = np.array(report["crystal_centre"])
    sv = np.array(scene["objects"][1]["shape"]["vertices"])
    assert scene["objects"][0]["name"] == "crystal"
    # in the solvent's footprint, and listed before it (priority order)
    assert sv[:, 0].min() < cx[0] < sv[:, 0].max()
    assert abs(cx[2]) < 1e-6


def test_droplet_rotates_with_the_loop_axis():
    """Regression: the old generator rotated the loop but not the droplet, so
    any non-default loop_axis left them non-coplanar."""
    sc = build_hampton_scene(solvent_volume_mm3=VOLUME, loop_axis=[0, -1, 0])
    rep = validate_scene(sc, requested_volume_mm3=VOLUME)
    sv = np.array(sc["objects"][1]["shape"]["vertices"])
    lp = np.array(sc["objects"][2]["shape"]["path"])
    # loop now extends in -y; the droplet must follow it
    assert lp[:, 1].min() < -0.4
    assert sv[:, 1].mean() < -0.1
    assert rep["rim_to_fiber_mm"]["max"] <= 0.020


def test_contact_angle_is_ignored():
    """With a pinned rim, contact angle is an output; passing one must not
    change the geometry (the old solver pretended to honour it)."""
    a = build_hampton_scene(solvent_volume_mm3=VOLUME, contact_angle_deg=10.0)
    b = build_hampton_scene(solvent_volume_mm3=VOLUME, contact_angle_deg=80.0)
    assert a["objects"][1]["shape"]["vertices"] == \
        b["objects"][1]["shape"]["vertices"]


def test_mitegen_scene_validates_without_a_droplet():
    sc = build_mitegen_scene(model="M2-L18SP-200")
    validate_scene(sc, expect_droplet=False)   # must not raise


# ---------------------------------------------------------------------------
# The validator catches deliberate corruption
# ---------------------------------------------------------------------------

def _corrupt(scene, fn):
    sc = copy.deepcopy(scene)
    fn(sc)
    return sc


def test_validator_catches_the_old_placement_bug(scene):
    """Translate the droplet back to the origin -- the exact defect the old
    generator shipped -- and the validator must name it."""
    def move_to_origin(sc):
        sv = np.array(sc["objects"][1]["shape"]["vertices"])
        _, c = __import__("crystal_harvester.droplet", fromlist=["x"]) \
            .mesh_volume_centroid(sv, np.array(sc["objects"][1]["shape"]["faces"]))
        sc["objects"][1]["shape"]["vertices"] = (sv - c).tolist()
    with pytest.raises(SceneValidationError, match="not pinned in the loop"):
        validate_scene(_corrupt(scene, move_to_origin),
                       requested_volume_mm3=VOLUME)


def test_validator_catches_a_wrong_volume(scene):
    def shrink(sc):
        sv = np.array(sc["objects"][1]["shape"]["vertices"])
        c = sv.mean(axis=0)
        sc["objects"][1]["shape"]["vertices"] = (c + 0.9 * (sv - c)).tolist()
    with pytest.raises(SceneValidationError, match="volume"):
        validate_scene(_corrupt(scene, shrink), requested_volume_mm3=VOLUME)


def test_validator_catches_a_one_sided_dome(scene):
    """The hemisphere fallback's signature: nothing below the loop plane."""
    def flatten(sc):
        sv = np.array(sc["objects"][1]["shape"]["vertices"])
        sv[:, 2] = np.abs(sv[:, 2])
        sc["objects"][1]["shape"]["vertices"] = sv.tolist()
    with pytest.raises(SceneValidationError, match="straddle"):
        validate_scene(_corrupt(scene, flatten), requested_volume_mm3=None)


def test_validator_catches_an_open_mesh(scene):
    def puncture(sc):
        sc["objects"][1]["shape"]["faces"] = \
            sc["objects"][1]["shape"]["faces"][:-1]
    with pytest.raises(SceneValidationError, match="not closed"):
        validate_scene(_corrupt(scene, puncture), requested_volume_mm3=VOLUME)


def test_validator_catches_a_degenerate_fallback_shape(scene):
    """The generator's other silent fallback was a sphere primitive (radius
    from volume, at the origin); any non-mesh solvent must be refused."""
    def sphere(sc):
        sc["objects"][1]["shape"] = {"type": "sphere",
                                     "centre": [0., 0., 0.], "radius": 0.078}
    with pytest.raises(SceneValidationError, match="not a surface_mesh"):
        validate_scene(_corrupt(scene, sphere), requested_volume_mm3=VOLUME)


def test_stem_is_glued_into_the_pin(scene):
    """The stem fibers must end inside the pin's metal (the break-face glue
    joint) -- the old generator left them floating 0.3 mm in front of it."""
    from crystal_harvester.validate import _point_in_csg
    pin = next(o for o in scene["objects"] if o["name"] == "pin")
    for name in ("stem_fiber_1", "stem_fiber_2"):
        stem = next(o for o in scene["objects"] if o["name"] == name)
        end = np.asarray(stem["shape"]["path"][-1], dtype=float)
        assert _point_in_csg(end, pin["shape"]), \
            f"{name} ends at {end} outside the pin"


def test_validator_catches_a_detached_stem(scene):
    """Truncate the stems back to the pin tip -- the old defect -- and the
    validator must name it."""
    def truncate(sc):
        for name in ("stem_fiber_1", "stem_fiber_2"):
            o = next(x for x in sc["objects"] if x["name"] == name)
            path = np.asarray(o["shape"]["path"], dtype=float)
            keep = path[path[:, 0] <= 0.70]
            o["shape"]["path"] = keep.tolist()
    with pytest.raises(SceneValidationError, match="not attached"):
        validate_scene(_corrupt(scene, truncate), requested_volume_mm3=VOLUME)


def test_validator_catches_inverted_priority(scene):
    def swap(sc):
        sc["objects"][0], sc["objects"][1] = sc["objects"][1], sc["objects"][0]
    with pytest.raises(SceneValidationError, match="priority"):
        validate_scene(_corrupt(scene, swap), requested_volume_mm3=VOLUME)
