"""Tests for the pre-computed frame library: the template pipeline that
serves every camera frame by transforming a rendered spindle sweep
instead of raytracing.

The claims here, and the test that guards each:

  * tracing is tile-independent -> test_tile_independence*
    (sizing the tile to fit the card must not change the pixels)
  * a template crop reproduces a live render -> test_template_matches_live*
    (guards the phi coupling: `ty` is lateral at phi=0, along the view
    axis at phi=90)
  * out-of-range requests are refused, not clamped -> test_*_raises
  * a library built with different parameters is stale -> test_is_current_*

Heavier tests are CUDA-gated and use a coarse sweep in tmp_path rather
than the committed frame_library/.
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


def _virtual_frame(man, lib_dir, rec):
    """The full rendered window, reconstituted from the tight crop on disk.

    Templates store only their content, so a test that wants to index the file
    in the VIRTUAL coordinates `pose_crop` speaks -- as the registration check
    below does -- has to paste it back onto the window first.  Cropping the file
    directly with a virtual box would silently read the wrong region, and PIL
    pads out-of-range boxes with black rather than raising, so it would fail as
    a plausible picture instead of an error.
    """
    from PIL import Image
    rnd = man["rendered"]
    window = (int(rnd["width"]), int(rnd["height"]))
    im = Image.open(os.path.join(lib_dir, rec["file"])).convert("RGB")
    origin = tuple(rec.get("content_origin_px", (0, 0)))
    if origin == (0, 0) and im.size == window:
        return im
    canvas = Image.new("RGB", window,
                       tuple(man.get("background_rgb") or (255, 255, 255)))
    canvas.paste(im, origin)
    return canvas


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
        tpl = np.asarray(_virtual_frame(man, lib_dir, rec)
                         .crop(wide)).astype(float)

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
                        quality=man["jpeg_quality"], format=man["format"],
                        psf=man["psf"])
    assert is_current(SCENE, lib_dir, **base)

    # `format` and `psf` matter as much as the geometric settings: a JPEG
    # library and a PNG one differ in compression loss, and a pre-PSF library
    # is geometrically sharper than the renderer now produces. Either mismatch
    # would serve pixels that no longer match a live render.
    for key, other in (("supersample", 4), ("step_deg", 1.0), ("n_cond", 7),
                       ("pan_mm", 99.0), ("axis", "roty"),
                       ("jpeg_quality", 60), ("format", "jpeg"),
                       ("psf", False)):
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
    assert resolved["format"] == sig["format"].default
    assert resolved["psf"] == sig["psf"].default
    # render_sha has no build_library parameter by design -- it is a property
    # of the code on disk, not a choice, so both sides must call the same
    # function rather than agree on a default.
    from loop_sim.library.frame_library import render_sha
    assert resolved["render_sha"] == render_sha()
    assert "render_sha" not in sig


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
# render_sha: the one staleness hole nothing else covered
# ---------------------------------------------------------------------------
def test_render_sha_covers_the_tracers_and_not_the_delivery_stage():
    """Getting this set wrong is silent BOTH ways -- too few files and a
    renderer change ships stale frames, too many and an unrelated edit costs
    hours of rebuild.  `field.py` is the one that must stay out: it runs at
    serve time, downstream of `pose_crop`, and never enters a template.  Being
    able to change it without a rebuild is why it was put there.
    """
    from loop_sim.library.frame_library import render_source_paths

    root, paths = render_source_paths()
    covered = {os.path.relpath(p, root).replace(os.sep, "/") for p in paths}
    for want in ("renderer/microscope.py", "renderer/engine_torch.py",
                 "renderer/optics.py", "motors/goniometer.py",
                 "scene/scene.py", "scene/tube.py", "scene/surface_mesh.py",
                 "scene/primitives.py", "scene/csg.py", "scene/materials.py"):
        assert want in covered, f"{want} decides template pixels but is not hashed"
    for keep_out in ("renderer/field.py", "renderer/pin_projection.py",
                     "renderer/torch_compat.py", "renderer/beam.py",
                     "renderer/xray_torch.py",
                     "library/frame_library.py", "server/camera_server.py"):
        assert keep_out not in covered, f"{keep_out} must not invalidate a library"


def test_render_sha_is_stable_and_content_sensitive(tmp_path):
    """Stable across calls (it is cached and compared on every status check),
    and sensitive to both content and file set."""
    from loop_sim.library.frame_library import _sha_over, render_sha

    assert render_sha() == render_sha()
    a, b = tmp_path / "a.py", tmp_path / "b.py"
    a.write_text("x = 1\n")
    b.write_text("y = 2\n")
    root = str(tmp_path)
    base = _sha_over(root, [str(a), str(b)])
    assert _sha_over(root, [str(a), str(b)]) == base
    a.write_text("x = 2\n")
    assert _sha_over(root, [str(a), str(b)]) != base, "content change not seen"
    a.write_text("x = 1\n")
    assert _sha_over(root, [str(a)]) != base, "a removed file was not seen"


def test_a_manifest_built_by_another_renderer_is_stale_not_current(tmp_path):
    """The gap this closes: `scene_sha256` catches a changed scene and the
    other build keys catch changed settings, but neither catches a
    renderer edit that leaves the manifest reading `current` while the
    frames were traced by code that no longer exists.
    """
    import json
    from loop_sim.library.frame_library import library_diff, render_sha

    man = dict(build_params(),
               frames=[{"file": "rot_0000.png", "angle": 0.0}],
               scene=SCENE)
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    assert library_diff(SCENE, str(tmp_path), **build_params()) == {}

    man["render_sha"] = "0" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    diff = library_diff(SCENE, str(tmp_path), **build_params())
    assert set(diff) == {"render_sha"}
    assert diff["render_sha"] == {"have": "0" * 64, "want": render_sha()}

    del man["render_sha"]                     # a manifest predating the key
    (tmp_path / "manifest.json").write_text(json.dumps(man))
    diff = library_diff(SCENE, str(tmp_path), **build_params())
    assert diff["render_sha"]["have"] is None


def test_render_sha_difference_reads_as_english_not_a_digest():
    """64 hex characters tell an operator nothing, and two of them tell them
    less.  The banner has to say what actually happened."""
    from loop_sim.server.camera_server import describe_differences

    changed = describe_differences({"render_sha": {"have": "a" * 64, "want": "b" * 64}})
    never = describe_differences({"render_sha": {"have": None, "want": "b" * 64}})
    assert "renderer" in changed and "a" * 8 not in changed
    assert "fingerprint" in never


def test_no_shipped_library_is_stale_on_build_parameters():
    """Adding a build key silently marks every shipped library stale, and the
    LAUNCH path rebuilds a stale library before it binds the socket -- so a
    bare `camera_server --scene ...` would start hours of work.  The three
    manifests carry the sha of the renderer that actually built them.

    Asserted on `library_diff`, NOT on `library_status`, because the two catch
    different things and only one of them is a mistake.  A build-PARAMETER
    mismatch means a key was added without stamping the manifests -- always a
    bug.  A `missing` status means the SCENE changed and a rebuild is owed,
    which is deliberate and is exactly the state `hampton_300um_realistic` is
    in after slice 3.  `library_diff` excludes the scene fingerprint by
    design, so it separates the two cleanly.
    """
    from loop_sim.library.frame_library import library_diff, load_manifest

    for name in ("hampton_300um", "hampton_300um_realistic", "mitegen_200um"):
        lib = os.path.join(REPO_ROOT, "frame_library", name)
        man = load_manifest(lib)
        if man is None:
            pytest.skip(f"{name} library not present")
        # graded against its OWN supersample, which is per-scene by design
        params = build_params(supersample=man["supersample"])
        path = os.path.join(REPO_ROOT, "scene_files", name + ".yaml")
        assert library_diff(path, lib, **params) == {}, name


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
    ap.add_argument("--template-format", choices=["png", "jpeg"], default=None)
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


@cuda_only
def test_template_source_delivers_on_the_sensor_raster(tiny_library):
    """`sensor` resamples the served frame onto the real camera's grid.

    The scene renders SQUARE pixels; the BL831 camera's are 1.11 non-square
    and it emits 704x480.  This is the one thing about the delivered frame
    that is not the scene's own resolution, and it must hold for every pose --
    a resample that only fired on some crops would change the magnification
    as the stage moved.
    """
    import io
    from PIL import Image
    from loop_sim.renderer.field import SENSOR_WH
    from loop_sim.server.camera_server import TemplateSource

    man, lib_dir = tiny_library
    src = TemplateSource(man, lib_dir, jpeg_quality=85, sensor=SENSOR_WH)
    for pose in ({}, {"rotx": 90.0}, {"tx": 0.2, "tz": 0.1, "rotx": 45.0}):
        assert Image.open(io.BytesIO(src.render(pose))).size == SENSOR_WH


@cuda_only
def test_servable_pose_is_exact_and_idempotent(tiny_library):
    """servable_pose must report the pose the library actually shows.

    Two properties matter.  A pose already in range must come back untouched
    (otherwise the readout drifts on every poll), and a clamped pose must
    itself be servable -- if it were not, the number shown would still not
    match the picture, which is the whole point.
    """
    from loop_sim.library.frame_library import servable_pose, pose_crop

    man, _lib_dir = tiny_library

    for p in ({"tx": 0.0, "ty": 0.0, "tz": 0.0, "angle_deg": 0.0, "zoom": 1.0},
              {"tx": 0.0, "ty": 0.0, "tz": 0.0, "angle_deg": 90.0, "zoom": 1.0}):
        out, note = servable_pose(man, **p)
        assert note is None, f"in-range pose reported as clamped: {note}"
        for k in ("tx", "ty", "tz", "zoom"):
            assert out[k] == pytest.approx(p[k], abs=1e-9), k

    # Absurd requests on every clamped axis; each result must be servable,
    # which pose_crop(clamp=False) asserts by not raising.
    for p in ({"tx": 0.0, "ty": 0.0, "tz": 0.0, "angle_deg": 0.0, "zoom": 999.0},
              {"tx": 0.0, "ty": 0.0, "tz": 0.0, "angle_deg": 0.0, "zoom": 1e-3},
              {"tx": 50.0, "ty": 0.0, "tz": 0.0, "angle_deg": 0.0, "zoom": 1.0},
              {"tx": 0.0, "ty": 9.0, "tz": 0.0, "angle_deg": 45.0, "zoom": 1.0}):
        out, note = servable_pose(man, **p)
        assert note is not None, f"out-of-range pose not reported: {p}"
        pose_crop(man, tx=out["tx"], ty=out["ty"], tz=out["tz"],
                  angle_deg=p["angle_deg"], zoom=out["zoom"], clamp=False)


@cuda_only
def test_rebuilding_in_another_format_removes_the_old_frames(tmp_path):
    """A format change must not leave the previous frames on disk.

    Frames are overwritten in place, so rot_0000.png does not replace
    rot_0000.jpg. Orphans are the bad kind of leftover: still on disk, still
    tracked by git (frame_library is a committed deliverable), referenced by no
    manifest -- so the library silently doubles in size and ships two copies.
    """
    root = str(tmp_path)
    build_library(SCENE, root=root, step_deg=90.0, supersample=1, n_cond=1,
                  format="jpeg", progress=None)
    lib_dir = library_dir(SCENE, root)
    assert [f for f in os.listdir(lib_dir) if f.endswith(".jpg")]

    man = build_library(SCENE, root=root, step_deg=90.0, supersample=1,
                        n_cond=1, format="png", progress=None)
    left = sorted(os.listdir(lib_dir))
    assert not [f for f in left if f.endswith(".jpg")], \
        f"stale JPEG frames survived the rebuild: {left}"
    assert all(f["file"].endswith(".png") for f in man["frames"])
    assert all(os.path.exists(os.path.join(lib_dir, f["file"]))
               for f in man["frames"])


# ---------------------------------------------------------------------------
# Template cache sizing: the fix for a slew missing on every frame
# ---------------------------------------------------------------------------
def _fake_manifest(w, h, n, crop=None):
    """A manifest just complete enough to size a cache from.

    `crop` gives each frame a `content_size_px`, which is what a real library
    carries now: the cache holds the decoded CROP, so sizing from the virtual
    window over-states a frame and under-fills the cache.
    """
    frames = [{"file": f"rot_{i:04d}.png"} for i in range(n)]
    if crop is not None:
        for f in frames:
            f["content_origin_px"] = [0, 0]
            f["content_size_px"] = list(crop)
    return {"rendered": {"width": w, "height": h}, "frames": frames}


def test_template_cache_sizes_from_the_stored_crop_not_the_window():
    """The cache holds what is DECODED, and that is the crop.

    Sizing from `rendered` was right while a template was the whole window and
    is a ~9x over-estimate now.  The error is silent and in the expensive
    direction: the cache comes out nine times smaller than the host can afford,
    so a slew keeps paying the decode this cache exists to remove.
    """
    from loop_sim.server.camera_server import plan_template_cache

    from loop_sim.server.camera_server import (_CACHE_RAM_FRACTION,
                                               _DECODED_BYTES_PER_PX)

    window = _fake_manifest(5578, 2570, 360)                     # 58 MiB/frame
    cropped = _fake_manifest(5578, 2570, 360, crop=(3940, 414))  # 6.6 MiB/frame
    # Small enough that neither result is clamped by the 360-frame library.
    avail = 2 * 2**30
    got = plan_template_cache(cropped, avail=avail)
    per = int(3940 * 414 * _DECODED_BYTES_PER_PX)
    assert got == int(avail * _CACHE_RAM_FRACTION) // per, \
        "the crop, not the window, is what a cached frame costs"
    assert got > 8 * plan_template_cache(window, avail=avail), \
        "8.8x smaller per frame must buy ~8.8x more of them"


def test_template_cache_holds_the_whole_library_when_ram_allows():
    """A slew visits every angle once per revolution, so anything smaller than
    the library misses on every rotating frame.  Given room, hold all of it."""
    from loop_sim.server.camera_server import plan_template_cache

    man = _fake_manifest(5578, 2570, 360)          # 41 MiB/frame, 14.4 GiB total
    assert plan_template_cache(man, avail=256 * 2**30) == 360, \
        "with 256 GiB available the whole sweep must fit"


def test_template_cache_shrinks_rather_than_promising_memory_it_lacks():
    """The failure this guards is a laptop trying to hold the sweep because the
    default said so.  Sized from AVAILABLE ram, never from total.

    `per` must be what a decoded frame REALLY costs -- PIL packs RGB into
    4-byte pixels, so `w*h*3` under-counts by a third and this function would
    over-promise, which is precisely backwards.
    """
    from loop_sim.server.camera_server import (plan_template_cache,
                                               _CACHE_RAM_FRACTION,
                                               _DECODED_BYTES_PER_PX)

    man = _fake_manifest(5578, 2570, 360)
    per = int(5578 * 2570 * _DECODED_BYTES_PER_PX)
    for avail_gib in (2, 8, 32):
        got = plan_template_cache(man, avail=avail_gib * 2**30)
        want = int((avail_gib * 2**30 * _CACHE_RAM_FRACTION) // per)
        assert got == max(1, min(360, want)), f"{avail_gib} GiB -> {got}"
        assert got >= 1, "never zero -- one frame must always be cacheable"
    # A tiny box still gets a working server, just a cold one.
    assert plan_template_cache(man, avail=64 * 2**20) == 1


def test_the_cache_never_plans_more_than_the_ram_it_was_given():
    """`plan_template_cache`'s one guarantee, tested against the guarantee:
    it must under-promise, returning fewer frames than the RAM could hold
    rather than more.  PIL decodes RGB into 4-byte-aligned RGBX, so a
    decoded frame costs `w*h*4`, not `w*h*3` -- see docs/DECISIONS.md
    2026-08-14 (decoded-template byte figures) for the measured figure.

    Checked arithmetically against a 4 B/px floor rather than by watching
    RSS: an allocator-based check inside a shared test process is not a
    measurement, because a previous test's freed arena can absorb the
    allocation and the delta reads zero.
    """
    from loop_sim.server.camera_server import (_CACHE_RAM_FRACTION,
                                               _DECODED_BYTES_PER_PX,
                                               plan_template_cache)

    assert _DECODED_BYTES_PER_PX >= 4.0, (
        "PIL packs RGB into 4-byte pixels; a constant below 4 makes the cache "
        "over-promise, which is the failure this whole function exists to avoid")

    for w, h, n in ((5578, 2570, 360), (3940, 414, 360), (1840, 2296, 360)):
        man = _fake_manifest(w, h, n, crop=(w, h))
        for avail_gib in (1, 2, 8, 32, 256):
            avail = avail_gib * 2**30
            planned = plan_template_cache(man, avail=avail)
            budget = avail * _CACHE_RAM_FRACTION
            # 4 B/px is the floor PIL can possibly use, so a plan that exceeds
            # the budget even at the floor is over-promising for certain.
            assert planned * w * h * 4 <= budget or planned == 1, (
                f"{w}x{h} at {avail_gib} GiB: planned {planned} frames = "
                f"{planned * w * h * 4 / 2**30:.2f} GiB against a "
                f"{budget / 2**30:.2f} GiB budget")


def test_template_cache_never_raises_on_a_malformed_manifest():
    """It runs in TemplateSource.__init__, so a throw here takes the server
    down at startup.  None means 'caller keeps its own default'."""
    from loop_sim.server.camera_server import plan_template_cache

    assert plan_template_cache({}) is None
    assert plan_template_cache({"rendered": {"width": 0, "height": 0},
                                "frames": []}) is None
    assert plan_template_cache(_fake_manifest(5578, 2570, 360), avail=0) is None


def test_template_cache_is_honoured_and_actually_stops_the_decode():
    """End to end on the real library: a big cache must turn a second pass over
    the same angles into cache hits, which is the entire 288 -> 69 ms effect."""
    from loop_sim.server.camera_server import TemplateSource

    lib_dir = os.path.join(REPO_ROOT, "frame_library", "hampton_300um")
    if not os.path.exists(os.path.join(lib_dir, "manifest.json")):
        pytest.skip("hampton_300um library not present")
    from loop_sim.library.frame_library import load_manifest
    man = load_manifest(lib_dir)

    # `cache_size=None` is still the conservative 8.  The SERVER now defaults to
    # "auto" -- affordable since templates became tight crops -- but a bare
    # TemplateSource must not claim memory nobody asked it for.
    assert TemplateSource(man, lib_dir)._cache_size == 8, \
        "cache_size=None must stay small; 'auto' is the server's default, not this"
    assert TemplateSource(man, lib_dir, cache_size="auto")._cache_size > 8, \
        "'auto' must size from RAM, not fall back to the default"
    with pytest.raises(ValueError):
        TemplateSource(man, lib_dir, cache_size="lots")

    src = TemplateSource(man, lib_dir, jpeg_quality=85, cache_size=16)
    assert src._cache_size == 16, "an explicit int must be honoured verbatim"
    angles = [0.0, 1.0, 2.0, 3.0, 4.0]
    for a in angles:
        src.render({man["axis"]: a})
    assert len(src._cache) == len(angles), "each distinct angle caches once"
    names = set(src._cache)
    for a in angles:                      # second pass: no new decodes
        src.render({man["axis"]: a})
    assert set(src._cache) == names, "a warm angle must not be re-decoded"

    # ...and the old 8-entry behaviour is still reachable, which is what
    # bench_serve --template-cache 8 uses to reproduce the pre-fix numbers.
    small = TemplateSource(man, lib_dir, jpeg_quality=85, cache_size=2)
    for a in (0.0, 1.0, 2.0):
        small.render({man["axis"]: a})
    assert len(small._cache) == 2, "LRU must still evict when told to"


# ---------------------------------------------------------------------------
# Tight crops: templates store their content, the reader fills the rest
# ---------------------------------------------------------------------------
def test_content_bbox_is_exact_and_survives_a_blank_frame():
    """The crop is only lossless if the bbox is exact rather than thresholded."""
    from loop_sim.library.frame_library import content_bbox, crop_to_content

    a = np.full((100, 200, 3), 255, np.uint8)
    assert content_bbox(a) is None, "an all-background frame has no content"
    # A blank frame must still store SOMETHING: PIL cannot hold a 0x0 image.
    blank, origin = crop_to_content(a, margin=16)
    assert blank.shape[:2] == (1, 1)

    a[40:60, 80:120] = 0
    assert content_bbox(a) == (80, 40, 119, 59)
    # One pixel one level off white is content, not noise.
    b = np.full((20, 20, 3), 255, np.uint8)
    b[5, 7] = (255, 254, 255)
    assert content_bbox(b) == (7, 5, 7, 5)

    cropped, (ox, oy) = crop_to_content(a, margin=16)
    assert (ox, oy) == (64, 24)
    assert cropped.shape[:2] == (52, 72)          # 20+32 rows, 40+32 cols
    # Clamped at the frame edge rather than running off it.
    _, origin = crop_to_content(a, margin=1000)
    assert origin == (0, 0)


def test_crop_margin_covers_the_readers_inward_rounding():
    """The reader rounds its output sub-rect inward, discarding up to one output
    pixel = `supersample / zoom_min` template px, and PIL's bilinear support
    reaches about as far again.  The margin has to cover both or the lowest
    servable zoom clips the sample."""
    from loop_sim.library.frame_library import crop_margin_px

    for supersample, zoom_min in ((4, 0.798), (1, 0.35), (8, 0.5)):
        assert crop_margin_px(supersample, zoom_min) >= 2.0 * supersample / zoom_min
    assert crop_margin_px(4, 4.0) == 16, "a floor, so the arithmetic is never tight"


def test_frames_complete_refuses_a_mismatched_library_both_ways():
    """The size check is the interlock between cropped and uncropped libraries.

    It is the only gate that returns "missing", the one verdict that
    actually blocks serving; a merely stale library is served with a
    warning.  It has to catch a manifest claiming a crop over full-window
    frames, and full-window frames under a manifest that declares none.
    """
    from PIL import Image
    from loop_sim.library.frame_library import _frames_complete

    import tempfile
    with tempfile.TemporaryDirectory() as d:
        Image.new("RGB", (40, 30), (255, 255, 255)).save(
            os.path.join(d, "rot_0000.png"))
        window = {"width": 100, "height": 80}

        # Declares the window, stores 40x30 -> a half-written or mis-declared
        # library, and must not serve.
        assert not _frames_complete(d, {"rendered": window,
                                        "frames": [{"file": "rot_0000.png"}]})
        # Declares the crop it actually stores -> fine.
        assert _frames_complete(d, {"rendered": window, "frames": [
            {"file": "rot_0000.png", "content_origin_px": [10, 10],
             "content_size_px": [40, 30]}]})
        # Declares a DIFFERENT crop -> must not serve.
        assert not _frames_complete(d, {"rendered": window, "frames": [
            {"file": "rot_0000.png", "content_origin_px": [10, 10],
             "content_size_px": [41, 30]}]})


def test_cropped_compose_reproduces_the_full_window_it_replaces():
    """THE claim the tight crop rests on, checked against the thing it replaced.

    Composing from a stored crop plus a background fill must give the same
    picture as resizing the whole rendered window -- because a template holds
    raw transmittance and every ray is born at 1.0, so what the crop omits is
    background exactly rather than approximately.

    One level of tolerance, not zero: PIL's resize coefficients are fixed-point,
    and the two constructions reach the same sample through different integer
    offsets.  Anything structurally wrong -- a dropped crop origin, an inward
    rounding that eats the sample, a sub-box clamp that rescales -- lands in the
    tens or hundreds of levels, not one.
    """
    from PIL import Image
    from loop_sim.library.frame_library import load_manifest
    from loop_sim.server.camera_server import TemplateSource

    lib_dir = os.path.join(REPO_ROOT, "frame_library", "hampton_300um_realistic")
    if not os.path.exists(os.path.join(lib_dir, "manifest.json")):
        pytest.skip("hampton_300um_realistic library not present")
    man = load_manifest(lib_dir)
    if not man["frames"][0].get("content_origin_px"):
        pytest.skip("library is not cropped")
    src = TemplateSource(man, lib_dir)
    zmin, zmax = zoom_limits(man)

    worst = 0
    for zoom in (round(zmin + 1e-6, 4), 1.0, 2.0, zmax):
        for ang in (0.0, 37.0, 90.0, 212.0):
            for tx, ty in ((0.0, 0.0), (0.3, 0.0), (0.0, -0.2)):
                rec = frame_for_angle(man, ang)
                box, out, _, _ = pose_crop(man, tx=tx, ty=ty, angle_deg=ang,
                                           zoom=zoom, clamp=True)
                new = np.asarray(src._compose(rec, box, out), np.int16)
                old = np.asarray(_virtual_frame(man, lib_dir, rec)
                                 .resize(out, Image.BILINEAR, box=box), np.int16)
                worst = max(worst, int(np.abs(new - old).max()))
    assert worst <= 1, f"cropped compose differs from the full window by {worst} levels"


def test_pil_sensor_stretch_matches_to_sensor():
    """The template path resamples 640->704 in PIL; the live path still uses
    `field.to_sensor`.  They must agree, or the same pose looks different
    depending on which engine served it.

    One level, because PIL rounds the interpolated result back to 8 bits where
    `to_sensor` keeps it in float.  The frame is 8-bit either way, so that is a
    rounding difference and not a loss -- but it has to stay one level.
    """
    from loop_sim.library.frame_library import load_manifest
    from loop_sim.renderer import field as F
    from loop_sim.server.camera_server import TemplateSource

    lib_dir = os.path.join(REPO_ROOT, "frame_library", "hampton_300um_realistic")
    if not os.path.exists(os.path.join(lib_dir, "manifest.json")):
        pytest.skip("hampton_300um_realistic library not present")
    man = load_manifest(lib_dir)
    src = TemplateSource(man, lib_dir, sensor=tuple(F.SENSOR_WH))
    zmin, zmax = zoom_limits(man)

    worst = 0
    for zoom in (round(zmin + 1e-6, 4), 1.0, zmax):
        for ang in (0.0, 90.0, 212.0):
            rec = frame_for_angle(man, ang)
            box, out, _, _ = pose_crop(man, angle_deg=ang, zoom=zoom, clamp=True)
            img = src._compose(rec, box, out)
            a = F.to_sensor(np.asarray(img, np.float64) / 255.0, F.SENSOR_WH)
            b = src._sensor_stretch(img)
            worst = max(worst, int(np.abs(a * 255 - b * 255).max()))
    assert worst <= 1, f"PIL stretch differs from to_sensor by {worst} levels"


def test_recrop_is_idempotent_and_leaves_the_library_current():
    """Cropping is a migration, not a build: it must not change a build key, so
    a cropped library stays `current` and nothing triggers an hours-long
    rebuild.  Running it twice must be a no-op."""
    import shutil
    import tempfile
    from loop_sim.library.frame_library import (build_params, is_current,
                                                load_manifest, recrop_library)

    lib_dir = os.path.join(REPO_ROOT, "frame_library", "mitegen_200um")
    if not os.path.exists(os.path.join(lib_dir, "manifest.json")):
        pytest.skip("mitegen_200um library not present")
    scene = os.path.join(REPO_ROOT, "scene_files", "mitegen_200um.yaml")
    before = is_current(scene, lib_dir, **build_params())

    with tempfile.TemporaryDirectory() as d:
        copy = os.path.join(d, "mitegen_200um")
        # Two frames is enough to prove the contract without copying 15 MB.
        os.makedirs(copy)
        man = load_manifest(lib_dir)
        man = dict(man, frames=man["frames"][:2])
        for f in man["frames"]:
            shutil.copy(os.path.join(lib_dir, f["file"]), copy)
        import json
        with open(os.path.join(copy, "manifest.json"), "w") as fh:
            json.dump(man, fh)

        out = recrop_library(copy, progress=None)
        assert all("content_origin_px" in f for f in out["frames"])
        sizes = [tuple(f["content_size_px"]) for f in out["frames"]]
        again = recrop_library(copy, progress=None)
        assert [tuple(f["content_size_px"]) for f in again["frames"]] == sizes, \
            "a second pass must not crop the crop"

    # The real library's staleness verdict is unchanged by any of this.
    assert is_current(scene, lib_dir, **build_params()) == before


def test_ensure_dynamo_binds_or_stubs_a_torch_that_lacks_it(monkeypatch):
    """Reproduces the voltron 2026-08-13 import failure and both recoveries.

    torch 2.0.1 leaves `torch._dynamo` unbound until something imports it, and
    `engine_torch` spells `@torch._dynamo.disable` in a class body -- so the
    whole GPU path was unimportable on the deployment machine while healthy in
    development.

    Driven against a fake torch rather than the real one: torch >= 2.1 serves
    `_dynamo` from a module-level `__getattr__`, so deleting the attribute does
    not reproduce 2.0.1, and mutating the real module risks leaving it broken
    for every test that runs after this one.
    """
    import builtins
    import types
    from loop_sim.renderer import torch_compat

    real_import = builtins.__import__

    def run(dynamo_import):
        fake = types.ModuleType("torch")          # a torch with no _dynamo
        monkeypatch.setitem(sys.modules, "torch", fake)
        monkeypatch.delitem(sys.modules, "torch._dynamo", raising=False)
        monkeypatch.setattr(builtins, "__import__", dynamo_import(fake))
        assert not hasattr(fake, "_dynamo"), "the 2.0.1 starting condition"
        return fake, torch_compat.ensure_dynamo()

    # 1. the submodule imports cleanly -> bound for real (plain torch 2.0.1)
    def importable(fake):
        def _imp(name, *a, **k):
            if name == "torch._dynamo":
                mod = types.ModuleType("torch._dynamo")
                mod.disable = lambda fn=None, **kw: fn
                fake._dynamo = mod
                sys.modules[name] = mod
                return fake
            return real_import(name, *a, **k)
        return _imp

    fake, ok = run(importable)
    assert ok is True and hasattr(fake, "_dynamo")

    # 2. the submodule raises -> stubbed (the pt env's Inductor pkg_resources
    #    bug, recorded in DECISIONS "TITAN V measured")
    def explodes(_fake):
        def _imp(name, *a, **k):
            if name == "torch._dynamo":
                raise ImportError("simulated Inductor pkg_resources failure")
            return real_import(name, *a, **k)
        return _imp

    fake, ok = run(explodes)
    assert ok is False, "the stub path must report that it stubbed"

    @fake._dynamo.disable                 # the bare spelling engine_torch uses
    def f(x):
        return x + 1
    assert f(1) == 2, "a stubbed disable must return a working function"
    assert fake._dynamo.disable()(lambda x: x * 2)(3) == 6, "called spelling too"
    assert fake._dynamo.maybe_mark_dynamic(None, 0) is None


def test_ensure_dynamo_is_a_noop_without_torch(monkeypatch):
    """The CPU reference path has no torch at all and must not be disturbed."""
    import builtins
    from loop_sim.renderer import torch_compat

    real_import = builtins.__import__

    def _imp(name, *a, **k):
        if name == "torch":
            raise ImportError("no torch here")
        return real_import(name, *a, **k)

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setattr(builtins, "__import__", _imp)
    assert torch_compat.ensure_dynamo() is False
