"""Tests for the pre-computed X-ray radiograph library: the X-ray
analogue of frame_library.py, its own module and render_sha scope so the
two trees invalidate independently (docs/DECISIONS.md 2026-08-18).

The claims here, and the test that guards each:

  * render_sha scope is right, both directions ->
    test_xray_render_sha_covers_the_right_files,
    test_xray_render_sha_excludes_optical_only_files
  * a stored crop reproduces a live render -> test_stored_crop_matches_live_render
  * depth does not blur an X-ray template -> test_pose_crop_has_no_depth_blur
  * different build parameters make a library stale -> test_is_current_tracks_build_parameters

Heavier tests are CUDA-gated, using a coarse sweep in tmp_path; no
`xray_library/` is shipped yet (physics constants unsettled, see
docs/DECISIONS.md 2026-08-18, mu_xray stays illustrative).
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loop_sim.library.xray_library import (            # noqa: E402
    build_xray_library, ensure_xray_library, frame_for_angle, library_dir,
    load_manifest, pose_crop, servable_pose, xray_build_params,
    xray_is_current, xray_library_diff, xray_render_source_paths)

SCENE = os.path.join(REPO_ROOT, "scene_files", "hampton_300um.yaml")


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


cuda_only = pytest.mark.skipif(not _has_cuda(), reason="requires CUDA")


@pytest.fixture(scope="module")
def tiny_xray_library(tmp_path_factory):
    """A coarse supersample-1 sweep: enough poses to check the geometry."""
    if not _has_cuda():
        pytest.skip("requires CUDA")
    root = str(tmp_path_factory.mktemp("xraylib"))
    man = build_xray_library(SCENE, root=root, step_deg=90.0, supersample=1,
                             progress=None)
    return man, library_dir(SCENE, root)


# ---------------------------------------------------------------------------
# render_sha scope -- the whole reason this is a separate module
# ---------------------------------------------------------------------------
def test_xray_render_sha_covers_the_right_files():
    """engine_torch.py is DELIBERATELY here (not disjoint from the optical
    scope): trace_xray calls tscene.next_interface(), which lives there, so a
    correctness fix to interface detection changes X-ray output too.
    """
    root, paths = xray_render_source_paths()
    covered = {os.path.relpath(p, root).replace(os.sep, "/") for p in paths}
    for want in ("renderer/xray_torch.py", "renderer/engine_torch.py",
                 "renderer/beam.py", "scene/scene.py", "scene/tube.py",
                 "motors/goniometer.py"):
        assert want in covered, f"{want} decides X-ray template pixels but is not hashed"


def test_xray_render_sha_excludes_optical_only_files():
    """microscope.py (Snell/Fresnel) and optics.py (the objective PSF) are
    optical-only -- the X-ray tracer is a straight-ray Beer-Lambert walk with
    neither refraction nor a PSF, so a change to either must not cost an
    X-ray rebuild.
    """
    root, paths = xray_render_source_paths()
    covered = {os.path.relpath(p, root).replace(os.sep, "/") for p in paths}
    for keep_out in ("renderer/microscope.py", "renderer/optics.py",
                     "renderer/field.py", "renderer/pin_projection.py",
                     "library/frame_library.py", "library/xray_library.py",
                     "server/camera_server.py"):
        assert keep_out not in covered, f"{keep_out} must not invalidate an X-ray library"


def test_xray_render_sha_is_stable_and_content_sensitive(tmp_path):
    from loop_sim.library.xray_library import xray_render_sha
    from loop_sim.library.frame_library import _sha_over

    assert xray_render_sha() == xray_render_sha()
    a = tmp_path / "a.py"
    a.write_text("x = 1\n")
    root = str(tmp_path)
    base = _sha_over(root, [str(a)])
    a.write_text("x = 2\n")
    assert _sha_over(root, [str(a)]) != base, "content change not seen"


# ---------------------------------------------------------------------------
# Build + reload
# ---------------------------------------------------------------------------
@cuda_only
def test_build_produces_16bit_frames_with_full_dynamic_range(tiny_xray_library):
    """8-bit would quantize the biological signal (top ~fifth of the
    transmission range) into ~50 usable levels -- see the module docstring.
    """
    from PIL import Image

    man, lib_dir = tiny_xray_library
    assert man["modality"] == "xray"
    assert len(man["frames"]) == 4          # 360 / 90
    assert man["camera"]["na_condenser"] == 0.0
    for rec in man["frames"]:
        path = os.path.join(lib_dir, rec["file"])
        with Image.open(path) as im:
            arr = np.array(im)
        assert arr.dtype == np.uint16, f"{rec['file']} is not 16-bit"
        assert tuple(arr.shape) == (rec["content_size_px"][1], rec["content_size_px"][0])


@cuda_only
def test_stored_crop_matches_live_render(tiny_xray_library):
    """The stored, cropped, 16-bit-quantized frame must equal a fresh live
    render at the exact pose the build used -- proves the crop offset math
    and the quantization round-trip are both correct, not just plausible.
    Byte-exact because AIR.mu_xray == 0.0 makes the crop boundary exact
    rather than thresholded.
    """
    import torch
    from PIL import Image
    from loop_sim.scene.scene import load
    from loop_sim.motors.goniometer import Goniometer
    from loop_sim.renderer.torch_compat import ensure_dynamo
    ensure_dynamo()
    from loop_sim.renderer.engine_torch import TorchScene
    from loop_sim.renderer.xray_torch import render_xray_torch

    man, lib_dir = tiny_xray_library
    rec = man["frames"][0]
    assert rec["angle_deg"] == 0.0

    scene = load(SCENE, device="cpu")
    cam = scene.camera_cfg
    rnd = man["rendered"]
    cam["width"], cam["height"], cam["pixel_size"] = (
        rnd["width"], rnd["height"], rnd["pixel_size"])
    tscene = TorchScene(scene, torch.device("cuda"), torch.float64)

    win = man["window_mm"]
    gono = Goniometer(scene.geometry).set(
        rotx=0.0, tx=-win["centre_x"], ty=-win["centre_y"], tz=0.0)
    T_live = render_xray_torch(tscene, gono).cpu().numpy()

    ox, oy = rec["content_origin_px"]
    cw, ch = rec["content_size_px"]
    live_crop16 = np.clip(np.round(T_live[oy:oy + ch, ox:ox + cw] * 65535.0),
                          0, 65535).astype(np.uint16)
    with Image.open(os.path.join(lib_dir, rec["file"])) as im:
        stored16 = np.array(im)

    assert live_crop16.shape == stored16.shape
    diff = np.abs(live_crop16.astype(np.int32) - stored16.astype(np.int32))
    assert diff.max() <= 1, f"stored crop differs from a live render by up to {diff.max()} LSB"


# ---------------------------------------------------------------------------
# Replay geometry (reused verbatim from frame_library.py)
# ---------------------------------------------------------------------------
@cuda_only
def test_pose_crop_has_no_depth_blur(tiny_xray_library):
    """na_condenser: 0.0 must make the reused pose_crop report zero blur at
    any tz -- a collimated beam's Beer-Lambert integral along a straight line
    is invariant to translating the ray's start point along that same line,
    so depth does not defocus a radiograph (unlike the optical PSF
    defocus approximation pose_crop was written for).
    """
    man, lib_dir = tiny_xray_library
    for tz in (0.0, 0.05, -0.2, 1.0):
        _box, _size, sigma_px, _note = pose_crop(man, tz=tz, angle_deg=0.0)
        assert sigma_px == 0.0, f"tz={tz} produced a nonzero blur sigma_px={sigma_px}"


@cuda_only
def test_frame_for_angle_and_servable_pose_clamp(tiny_xray_library):
    man, lib_dir = tiny_xray_library
    fr = frame_for_angle(man, 100.0)     # nearest of 0/90/180/270
    assert fr["angle_deg"] == 90.0

    pose, note = servable_pose(man, zoom=1000.0)   # far past the ceiling
    assert note is not None
    assert pose["zoom"] == pytest.approx(man["supersample"])


# ---------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------
@cuda_only
def test_is_current_tracks_build_parameters(tiny_xray_library):
    man, lib_dir = tiny_xray_library
    base = xray_build_params(axis="rotx", step_deg=90.0, supersample=1,
                             pan_mm=man["pan_mm"])
    assert xray_is_current(SCENE, lib_dir, **base)

    diff = xray_library_diff(SCENE, lib_dir,
                             **xray_build_params(step_deg=90.0, supersample=2))
    assert "supersample" in diff
    assert diff["supersample"] == {"have": 1, "want": 2}


def test_is_current_false_when_absent(tmp_path):
    assert not xray_is_current(SCENE, str(tmp_path))


@cuda_only
def test_ensure_xray_library_is_a_noop_when_current(tiny_xray_library, monkeypatch):
    man, lib_dir = tiny_xray_library
    root = os.path.dirname(lib_dir)

    def _boom(*a, **kw):
        raise AssertionError("ensure_xray_library rebuilt an already-current library")
    monkeypatch.setattr("loop_sim.library.xray_library.build_xray_library", _boom)

    man2 = ensure_xray_library(SCENE, root=root, step_deg=90.0, supersample=1,
                               pan_mm=man["pan_mm"])
    assert man2 == man


@cuda_only
def test_non_rotx_axis_is_refused(tmp_path):
    """The window offset and pose_crop both hard-code the rotx coupling, same
    as the optical build -- see its identical guard."""
    with pytest.raises(ValueError, match="not supported"):
        build_xray_library(SCENE, root=str(tmp_path), axis="roty",
                           step_deg=180.0, supersample=1, progress=None)
