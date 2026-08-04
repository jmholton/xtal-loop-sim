"""
Tests for the pre-computed frame library: the template pipeline that serves
every camera frame by transforming a rendered spindle sweep instead of
raytracing.

The load-bearing claims, and the test that guards each:

  * tracing is tile-independent            -> test_tile_independence*
    (the whole bounded-VRAM strategy rests on this; without it, sizing the
    tile to fit the card would change the pixels)
  * a template crop reproduces a live render -> test_template_matches_live*
    (guards the phi coupling and the crop signs: the stage rides on the
    spindle, so `ty` is lateral at phi=0 and along the view axis at phi=90)
  * out-of-range requests are refused, not clamped -> test_*_raises
  * a library built with different parameters is stale -> test_is_current_*

Building a library renders, so the heavier tests are CUDA-gated and use a
coarse sweep in tmp_path rather than the committed frame_library/.

Run:  pytest tests/test_frame_library.py -v
"""
import inspect
import math
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loop_sim.library.frame_library import (            # noqa: E402
    _round_up_to_parity, build_library, build_params, frame_for_angle,
    is_current, library_dir, plan_window, pose_crop, zoom_limits)

SCENE = os.path.join(REPO_ROOT, "scene_files", "hampton_300um.yaml")


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


cuda_only = pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")


@pytest.fixture(scope="module")
def tiny_library(tmp_path_factory):
    """A coarse supersample-1 sweep: enough poses to check the geometry."""
    if not _has_cuda():
        pytest.skip("requires CUDA")
    root = str(tmp_path_factory.mktemp("lib"))
    man = build_library(SCENE, root=root, step_deg=90.0, supersample=1,
                        n_cond=1, quality=95, progress=None)
    return man, library_dir(SCENE, root)


# ---------------------------------------------------------------------------
# Tile independence -- the assumption the VRAM strategy rests on
# ---------------------------------------------------------------------------
@cuda_only
def test_tile_independence_is_byte_exact():
    """Splitting the trace into arbitrary chunks must not change a single bit.

    Tiles are sized at runtime to fit free VRAM, so if results depended on the
    tile the rendered image would depend on how busy the GPU happened to be.
    """
    import torch
    from loop_sim.scene.scene import load
    from loop_sim.renderer.engine_torch import TorchScene, render_torch
    from loop_sim.motors.goniometer import Goniometer

    scene = load(SCENE, device="cpu")
    scene.camera_cfg["width"], scene.camera_cfg["height"] = 160, 120
    tscene = TorchScene(scene, torch.device("cuda"), torch.float64)
    gono = Goniometer(scene.geometry).set(rotx=37.0)

    ref = render_torch(tscene, gono, n_cond=7, tile_size=10 ** 9)
    # deliberately awkward tiles, including ones that straddle the boundary
    # between condenser samples
    for tile in (19_200, 7_919, 1_000):
        got = render_torch(tscene, gono, n_cond=7, tile_size=tile)
        assert torch.equal(ref, got), f"tile_size={tile} changed the image"


@cuda_only
def test_auto_tile_stays_within_budget_and_is_exact():
    import torch
    from loop_sim.scene.scene import load
    from loop_sim.renderer.engine_torch import TorchScene, render_torch

    from loop_sim.motors.goniometer import Goniometer

    scene = load(SCENE, device="cpu")
    scene.camera_cfg["width"], scene.camera_cfg["height"] = 320, 240
    tscene = TorchScene(scene, torch.device("cuda"), torch.float64)
    gono = Goniometer(scene.geometry).set(rotx=15.0)

    ref = render_torch(tscene, gono, n_cond=3, tile_size=10 ** 9)
    for frac in (0.80, 0.10):
        got = render_torch(tscene, gono, n_cond=3, tile_size=None,
                           vram_fraction=frac)
        assert torch.equal(ref, got), f"auto tile at vram_fraction={frac} changed the image"


@cuda_only
def test_plan_tile_size_returns_a_tile_that_actually_fits():
    """The chosen tile must be one that was measured, not extrapolated to.

    Two regressions this guards. (1) Fitting a slope from two tiny probes and
    solving for a ~10M-ray tile is a 150x extrapolation; on the mesh path,
    whose Moller-Trumbore temporaries fragment the pool, it under-predicted by
    ~2.7 GB and pushed the render past the card. (2) The doubling ramp that
    replaced it first sampled through one fixed stride, so every rung past the
    subsample's length re-measured the same rays, reported a flat cost, and
    doubled all the way to the top -- which then tried an 87 GiB allocation.
    """
    import torch
    from loop_sim.scene.scene import load
    from loop_sim.renderer.engine_torch import TorchScene, plan_tile_size
    from loop_sim.renderer.microscope import _condenser_offsets  # noqa: F401
    from loop_sim.motors.goniometer import Goniometer, apply_transform

    scene = load(SCENE, device="cpu")
    scene.camera_cfg["width"], scene.camera_cfg["height"] = 900, 700
    tscene = TorchScene(scene, torch.device("cuda"), torch.float64)

    cam, g = scene.camera_cfg, scene.geometry
    W, H = int(cam["width"]), int(cam["height"])
    px = float(cam["pixel_size"])
    fast = np.array(g.get("camera_fast", [1, 0, 0]), float)
    slow = np.array(g.get("camera_slow", [0, 1, 0]), float)
    oax = np.array(g.get("optical_axis", [0, 0, -1]), float)
    oax /= np.linalg.norm(oax)
    xs = (np.arange(W) - W / 2.0) * px
    ys = (np.arange(H) - H / 2.0) * px
    gx, gy = np.meshgrid(xs, ys)
    focal = (gx[:, :, None] * fast + gy[:, :, None] * slow).reshape(-1, 3)
    Ti = Goniometer(g).set(rotx=23.0).transform_inv()
    focal_s = apply_transform(Ti, focal)
    oax_s = Ti[:3, :3] @ oax
    oax_s /= np.linalg.norm(oax_s) + 1e-30
    o = torch.as_tensor(focal_s - 50.0 * oax_s, device="cuda", dtype=torch.float64)
    d = torch.as_tensor(np.broadcast_to(oax_s, (W * H, 3)).copy(),
                        device="cuda", dtype=torch.float64)

    free_before, _ = torch.cuda.mem_get_info()
    tile = plan_tile_size(tscene, o, d, float(cam["na_objective"]), oax_s,
                          W * H, vram_fraction=0.5)
    assert 0 < tile <= W * H

    # Tracing at the chosen tile must stay inside the budget it was sized for.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_reserved()
    tscene.trace_rays(o[:tile], d[:tile], float(cam["na_objective"]), oax_s)
    torch.cuda.synchronize()
    cost = torch.cuda.max_memory_reserved() - before
    assert cost <= free_before * 0.5 * 1.25, (
        f"tile of {tile} rays cost {cost/2**30:.2f} GiB against a "
        f"{free_before*0.5/2**30:.2f} GiB budget -- the sizer over-committed")


def test_plan_tile_size_is_a_noop_on_cpu():
    """No device memory to budget, so the whole ray set goes in one pass."""
    import torch
    from loop_sim.scene.scene import load
    from loop_sim.renderer.engine_torch import TorchScene, plan_tile_size

    scene = load(SCENE, device="cpu")
    tscene = TorchScene(scene, torch.device("cpu"), torch.float64)
    assert plan_tile_size(tscene, None, None, 0.1, None, 12345) == 12345


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------
def test_round_up_to_parity():
    assert _round_up_to_parity(1395, 640) == 1396      # odd -> even
    assert _round_up_to_parity(1396, 640) == 1396      # already even
    assert _round_up_to_parity(10, 7) == 11            # even -> odd to match


def test_plan_window_covers_content_and_the_centred_view():
    """The window must hold both the whole sample and the home field of view.

    A mount is long and thin -- the hampton pin reaches x=6.7 mm against a
    4.7 mm field -- so a window centred on the origin leaves most of the pin
    unrendered and panning scrolls in blank background.
    """
    cam = {"width": 640, "height": 480, "pixel_size": 0.0074}
    content = (-0.06, 6.75, -0.30, 0.36)
    x0, x1, y0, y1 = plan_window(cam, content, pan_mm=0.6)

    assert x0 <= content[0] and x1 >= content[1]        # holds the sample
    assert y0 <= content[2] and y1 >= content[3]
    fw, fh = 640 * 0.0074 / 2, 480 * 0.0074 / 2
    assert x0 <= -fw and x1 >= fw                       # holds the centred FOV
    assert y0 <= -fh and y1 >= fh
    assert x1 - x0 > content[1] - content[0]            # plus pan headroom


# ---------------------------------------------------------------------------
# The pose -> crop transform
# ---------------------------------------------------------------------------
@cuda_only
def test_template_matches_live_render(tiny_library):
    """A template crop must reproduce what the renderer would have produced.

    This is the test that catches a wrong phi coupling or an inverted crop
    sign: both leave the picture looking perfectly plausible while showing the
    wrong part of the sample. Poses are chosen so that at phi=90 the roles of
    Y and Z swap, which is exactly where a naive `ty` crop is a no-op.
    """
    import torch
    from PIL import Image
    from loop_sim.scene.scene import load
    from loop_sim.renderer.engine_torch import TorchScene, render_torch
    from loop_sim.motors.goniometer import Goniometer

    man, lib_dir = tiny_library
    px = float(man["camera"]["pixel_size"])
    scene = load(SCENE, device="cpu")
    tscene = TorchScene(scene, torch.device("cuda"), torch.float64)

    poses = [
        {},
        {"tx": 10 * px},
        {"tx": -20 * px},
        {"ty": 10 * px},
        {"rotx": 90.0},
        {"rotx": 90.0, "tz": 10 * px},
        {"rotx": 180.0, "ty": -15 * px},
        {"rotx": 270.0, "tz": -9 * px},
    ]
    pad = 3
    for pose in poses:
        gono = Goniometer(scene.geometry).set(**pose)
        live = (render_torch(tscene, gono, n_cond=1) * 255).clamp(0, 255)
        live = live.to(torch.uint8).cpu().numpy().astype(float)

        angle = pose.get("rotx", 0.0)
        rec = frame_for_angle(man, angle)
        box, _, _, _ = pose_crop(man, tx=pose.get("tx", 0.0), ty=pose.get("ty", 0.0),
                                 tz=pose.get("tz", 0.0), angle_deg=angle, zoom=1.0)
        box = tuple(int(round(b)) for b in box)
        wide = (box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad)
        tpl = np.asarray(Image.open(os.path.join(lib_dir, rec["file"]))
                         .convert("RGB").crop(wide)).astype(float)

        H, W = live.shape[:2]
        best = min((np.abs(tpl[pad + dy:pad + dy + H, pad + dx:pad + dx + W] - live).mean(),
                    dx, dy)
                   for dy in range(-pad, pad + 1) for dx in range(-pad, pad + 1))
        assert (best[1], best[2]) == (0, 0), (
            f"pose {pose} best-matches the live render at a shift of "
            f"({best[1]},{best[2]}) px -- the crop is misaligned")


@cuda_only
def test_defocus_follows_the_spindle(tiny_library):
    """Depth is whichever translation currently points along the view axis.

    Because the stage rides on the spindle, Y is lateral at phi=0 and along
    the view axis at phi=90; Z is the reverse. The blur has to follow.
    """
    man, _ = tiny_library
    _, _, sig_y0, _ = pose_crop(man, ty=0.3, angle_deg=0.0)
    _, _, sig_y90, _ = pose_crop(man, ty=0.3, angle_deg=90.0)
    _, _, sig_z0, _ = pose_crop(man, tz=0.3, angle_deg=0.0)
    _, _, sig_z90, _ = pose_crop(man, tz=0.3, angle_deg=90.0)

    assert sig_y0 == pytest.approx(0.0, abs=1e-9)    # Y is lateral here
    assert sig_y90 > 0.5                             # ...and depth here
    assert sig_z0 > 0.5                              # Z is depth here
    assert sig_z90 == pytest.approx(0.0, abs=1e-9)   # ...and lateral here


@cuda_only
def test_zoom_outside_the_servable_range_raises(tiny_library):
    """Above the supersample there is no detail left; below the zoom floor the
    field of view is wider than the rendered window."""
    man, _ = tiny_library
    zmin, zmax = zoom_limits(man)
    pose_crop(man, zoom=zmax)                         # exactly at the ceiling: fine
    pose_crop(man, zoom=zmin)                         # exactly at the floor: fine
    with pytest.raises(ValueError, match="outside the"):
        pose_crop(man, zoom=zmax * 2)
    with pytest.raises(ValueError, match="outside the"):
        pose_crop(man, zoom=zmin * 0.5)


@cuda_only
def test_clamping_never_distorts_the_image(tiny_library):
    """Clamping must slide the box, never squeeze it.

    Squeezing the two axes independently changes magnification per axis, so the
    served frame is stretched -- and it looks entirely plausible on screen.
    """
    man, _ = tiny_library
    rnd, S = man["rendered"], man["supersample"]
    zmin, _ = zoom_limits(man)
    for pose in ({"tx": 500.0}, {"ty": -500.0}, {"zoom": zmin * 0.1},
                 {"tx": 500.0, "ty": 500.0, "zoom": zmin * 0.5}):
        box, out, _, note = pose_crop(man, clamp=True, **pose)
        assert note, f"clamped {pose} without reporting it"
        assert box[0] >= -1e-6 and box[1] >= -1e-6
        assert box[2] <= rnd["width"] + 1e-6 and box[3] <= rnd["height"] + 1e-6
        # aspect preserved: source box must match the camera's aspect exactly
        assert ((box[2] - box[0]) / (box[3] - box[1])
                == pytest.approx(out[0] / out[1], rel=1e-9))


@cuda_only
def test_crop_span_is_exact_and_zoom_scales_it(tiny_library):
    """The source span must be exactly W*supersample/zoom template pixels.

    Rounding it to integers makes magnification flicker by ~0.1% as a pan
    crosses pixel boundaries, because the result is always resized to W.
    """
    man, _ = tiny_library
    W = man["camera"]["width"]
    S = man["supersample"]
    for zoom in (1.0, 1.7, 2.0, float(S)):
        if zoom > S:
            continue
        box, _, _, _ = pose_crop(man, zoom=zoom)
        assert (box[2] - box[0]) == pytest.approx(W * S / zoom, rel=1e-9)
    wide, _, _, _ = pose_crop(man, zoom=zoom_limits(man)[0])
    narrow, _, _, _ = pose_crop(man, zoom=float(S))
    assert (wide[2] - wide[0]) > (narrow[2] - narrow[0])


def test_pose_crop_registration_offset_is_half_a_source_pixel():
    """PIL maps output pixel i to source EDGE left + (i+0.5)*scale, so the box
    origin sits half a source pixel before the first sample point.

    Without this the served image is offset by 0.5*(scale-1) template px --
    0.375 camera px at 4x, invisible to an integer-shift comparison.
    """
    man = {
        "camera": {"width": 640, "height": 480, "pixel_size": 0.0074,
                   "na_condenser": 0.07},
        "rendered": {"width": 5578, "height": 2570, "pixel_size": 0.0074 / 4},
        "window_mm": {"centre_x": 0.0, "centre_y": 0.0},
        "supersample": 4,
    }
    box, _, _, _ = pose_crop(man, zoom=1.0)
    scale = 4.0
    expected_left = 5578 / 2.0 - 640 * scale / 2.0 - 0.5 * scale + 0.5
    assert box[0] == pytest.approx(expected_left, abs=1e-9)
    # at zoom == supersample the scale is 1 and the offset vanishes
    box1, _, _, _ = pose_crop(man, zoom=4.0)
    assert box1[0] == pytest.approx(5578 / 2.0 - 640 / 2.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------
@cuda_only
def test_is_current_tracks_build_parameters(tiny_library):
    """Asking for different settings and silently getting the old library back
    would be indistinguishable from a correct build."""
    man, lib_dir = tiny_library
    base = build_params(axis="rotx", step_deg=90.0, supersample=1,
                        pan_mm=man["pan_mm"], n_cond=1,
                        quality=man["jpeg_quality"])
    assert is_current(SCENE, lib_dir, **base)

    for key, other in (("supersample", 4), ("step_deg", 1.0), ("n_cond", 7),
                       ("pan_mm", 99.0), ("axis", "roty"),
                       ("jpeg_quality", 60)):
        assert not is_current(SCENE, lib_dir, **dict(base, **{key: other})), \
            f"a change to {key} should invalidate the library"


def test_build_params_matches_build_library_defaults():
    """These two drifting apart makes every default build look stale, or every
    changed build look current."""
    sig = inspect.signature(build_library).parameters
    resolved = build_params()
    assert resolved["axis"] == sig["axis"].default
    assert resolved["step_deg"] == sig["step_deg"].default
    assert resolved["n_cond"] == sig["n_cond"].default
    assert resolved["supersample"] == sig["supersample"].default
    assert resolved["pan_mm"] == sig["pan_mm"].default
    assert resolved["jpeg_quality"] == sig["quality"].default


@cuda_only
def test_non_rotx_axis_is_refused(tmp_path):
    """The window offset and pose_crop both hard-code the rotx coupling, so
    another axis would silently produce a geometrically wrong library."""
    with pytest.raises(ValueError, match="not supported"):
        build_library(SCENE, root=str(tmp_path), axis="roty", step_deg=180.0,
                      supersample=1, n_cond=1, progress=None)


def test_is_current_false_when_absent(tmp_path):
    assert not is_current(SCENE, str(tmp_path))


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------
def test_server_defaults_do_not_invalidate_a_default_library():
    """Launching the server with no flags must NOT look stale to a library
    built with no flags.

    Regression: `--jpeg-quality` (the quality the server SENDS, default 85) was
    forwarded as the library's `quality` (the quality of the STORED template,
    default 90). Every default launch therefore disagreed with the shipped
    library and silently kicked off a 45-minute rebuild. Both end-to-end tests
    missed it because they passed explicit library_kwargs instead of going
    through the CLI.
    """
    import argparse
    from loop_sim.server.camera_server import library_kwargs_from_args

    # The real CLI defaults, taken from the real parser rather than restated.
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-cond", type=int, default=7)
    ap.add_argument("--jpeg-quality", type=int, default=85)
    ap.add_argument("--supersample", type=int, default=None)
    ap.add_argument("--template-quality", type=int, default=None)
    args = ap.parse_args([])

    # What is_current() will compare against the manifest, vs what a no-flag
    # `python -m loop_sim.library` writes into it.
    requested = build_params(**library_kwargs_from_args(args))
    default_build = build_params()
    differing = {k: (requested[k], default_build[k])
                 for k in requested if requested[k] != default_build[k]}
    assert not differing, (
        f"launching the server with no flags would rebuild a default-built "
        f"library; disagreeing parameters (requested, built): {differing}")


@cuda_only
def test_template_source_serves_without_a_gpu(tiny_library):
    """Serving is pure image work: decode, crop, scale, blur, encode."""
    import io
    from PIL import Image
    from loop_sim.server.camera_server import TemplateSource

    man, lib_dir = tiny_library
    src = TemplateSource(man, lib_dir, jpeg_quality=85)
    for pose in ({}, {"rotx": 90.0}, {"tx": 0.2, "tz": 0.1, "rotx": 45.0},
                 {"zoom": 1.0}):
        jpeg = src.render(pose)
        img = Image.open(io.BytesIO(jpeg))
        assert img.size == (man["camera"]["width"], man["camera"]["height"])
