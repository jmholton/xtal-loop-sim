"""Camera emulation: the two measured anchors, determinism, and the tone gate.

`loop_sim/renderer/field.py` maps the tracer's transmittance to what the camera
would record.  These tests pin the properties the rest of the delivery chain
depends on -- especially DETERMINISM, because `test_server_settle_parity`
compares JPEG bytes between the live server and a fresh render, and any RNG or
per-call state in the camera stage would break that guard rather than this one.
"""
import os
import sys

import numpy as np
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from loop_sim.renderer import field as F

H, W = 96, 128


def _t(value):
    return np.full((H, W, 3), float(value))


# --- the two measured anchors ---------------------------------------------

def test_clear_path_reads_the_empty_field():
    """T = 1 must reproduce the measured empty field, not pure white."""
    out = F.apply_camera(_t(1.0))
    assert out.mean() == pytest.approx(F.E0, rel=2e-3)
    assert out.max() < 1.0                      # never reaches the white rail


def test_opaque_reads_the_black_floor():
    """T = 0 must reproduce the measured pin interior, not pure black."""
    out = F.apply_camera(_t(0.0))
    assert np.allclose(out, F.B)
    assert out.min() > 0.0                      # never reaches the black rail


def test_neither_rail_is_reachable_for_any_transmittance():
    t = np.linspace(0.0, 1.0, H * W * 3).reshape(H, W, 3)
    out = F.apply_camera(t)
    q = np.round(out * 255).astype(int)
    assert q.min() > 0 and q.max() < 255


# --- determinism: load-bearing for the byte comparisons --------------------

def test_repeated_calls_are_bit_identical():
    t = np.linspace(0.0, 1.0, H * W * 3).reshape(H, W, 3)
    a = F.apply_camera(t)
    b = F.apply_camera(t.copy())
    assert np.array_equal(a, b)


def test_field_does_not_depend_on_call_order_or_cache_state():
    """The shape cache must be a pure memo, never observable in the output."""
    first = F.vignette(H, W).copy()
    F.vignette(H * 2, W * 2)                    # perturb the cache
    F.vignette(H, W, amplitude=0.3)
    assert np.array_equal(F.vignette(H, W), first)


# --- the field shape -------------------------------------------------------

def test_field_shape_has_unit_mean_and_is_resolution_independent():
    """Normalised coordinates mean a template built at one size and served at
    another sees the same physical field."""
    small = F.vignette(120, 176)
    large = F.vignette(480, 704)
    assert small.mean() == pytest.approx(1.0, abs=0.02)
    assert large.mean() == pytest.approx(1.0, abs=0.02)
    assert large.min() == pytest.approx(small.min(), abs=0.02)
    assert large.max() == pytest.approx(small.max(), abs=0.02)


def test_amplitude_zero_is_a_flat_field():
    """Recent epochs are 5-7x flatter than the calibration session, so the
    amplitude dial has to reach flat."""
    assert np.allclose(F.vignette(H, W, amplitude=0.0), 1.0)


def test_field_is_a_vertical_bowl():
    """The dominant term is v^2: bright through the middle rows, falling off
    top and bottom.  A plane explains under 1% of this."""
    f = F.vignette(H, W)
    rows = f.mean(axis=1)
    assert rows[H // 2] > rows[0] and rows[H // 2] > rows[-1]


# --- the sensor raster -----------------------------------------------------

def test_sensor_resample_lands_on_the_camera_raster():
    """The tracer renders square pixels; the BL831 camera's are 1.11
    non-square and it emits 704x480."""
    out = F.to_sensor(np.zeros((480, 640, 3)))
    assert out.shape == (F.SENSOR_WH[1], F.SENSOR_WH[0], 3)


def test_sensor_resample_is_the_identity_at_the_target_size():
    t = np.linspace(0.0, 1.0, 480 * 704 * 3).reshape(480, 704, 3)
    assert np.array_equal(F.to_sensor(t), t)


def test_sensor_resample_is_bit_identical_across_calls():
    """Load-bearing for `test_server_settle_parity`, which compares JPEG
    bytes between the live server and a fresh render."""
    t = np.linspace(0.0, 1.0, 480 * 640 * 3).reshape(480, 640, 3)
    assert np.array_equal(F.to_sensor(t), F.to_sensor(t.copy()))


def test_sensor_resample_preserves_a_flat_field_exactly():
    """Bilinear weights must sum to 1 on every output column, including the
    two clamped edges -- otherwise the frame darkens at its own border."""
    out = F.to_sensor(np.full((480, 640, 3), 0.42))
    assert np.allclose(out, 0.42, atol=1e-12)


def test_sensor_resample_does_not_shift_the_image():
    """A centred feature must stay centred.  Aligning corners instead of
    pixel centres would slide it half an output pixel."""
    t = np.zeros((480, 640, 3))
    t[:, 320 - 40:320 + 40] = 1.0
    out = F.to_sensor(t)
    col = out[240, :, 0]
    centroid = float((np.arange(col.size) * col).sum() / col.sum())
    assert centroid == pytest.approx((704 - 1) / 2.0, abs=0.05)


def test_sensor_resample_stretches_the_horizontal_axis_only():
    """640 -> 704 is 1.100x, which is how the 1.110 pixel aspect is
    reproduced.  The vertical pitch is already the camera's."""
    t = np.zeros((480, 640, 3))
    t[200:280, 280:360] = 1.0                   # an 80x80 square
    out = F.to_sensor(t)
    wide = (out[240, :, 0] > 0.5).sum()
    tall = (out[:, 352, 0] > 0.5).sum()
    assert wide == pytest.approx(80 * 704 / 640, abs=1)
    assert tall == 80


# --- mono ------------------------------------------------------------------

def test_mono_collapses_channel_spread():
    """Material colour is an absorption spectrum here, so a 'blue' crystal
    renders blue.  mono masks that until the scene YAML is fixed."""
    t = np.zeros((H, W, 3))
    t[..., 0], t[..., 1], t[..., 2] = 0.2, 0.6, 0.9
    colour = F.apply_camera(t, mono=False)
    grey = F.apply_camera(t, mono=True)
    assert colour.std(axis=2).max() > 0.05
    assert grey.std(axis=2).max() == pytest.approx(0.0, abs=1e-12)


# --- the off switch --------------------------------------------------------

def test_disabled_is_the_identity():
    t = np.linspace(0.0, 1.0, H * W * 3).reshape(H, W, 3)
    assert F.apply_camera(t, enabled=False) is t


# --- the acceptance gate ---------------------------------------------------

def test_tone_gate_binary_render_becomes_continuous_tone():
    """The gap this module exists to close.  A raytraced frame is ~85% pure
    white and ~14% pure black; real BL831 frames carry 19-27% intermediate
    tone.  Simulate the binary case and require the stage to fix it.
    """
    t = np.zeros((H, W, 3))
    t[:, : W // 2] = 1.0                        # a hard silhouette edge
    before = np.round(t * 255).astype(int)
    assert np.mean((before > 0) & (before < 255)) < 0.01

    after = np.round(F.apply_camera(t) * 255).astype(int)
    assert np.mean((after > 0) & (after < 255)) >= 0.15
