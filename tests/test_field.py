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


# --- the pin's specular streak ---------------------------------------------

PIN_H, PIN_W = 480, 704


def _pin_frame(width=95, tilt=0.04, x_from=300):
    """A transmittance frame holding one opaque bar, like a pin in view."""
    yy, xx = np.mgrid[0:PIN_H, 0:PIN_W].astype(float)
    bar = (np.abs((yy - 240) - tilt * (xx - 350)) < width / 2.0) & (xx > x_from)
    t = np.ones((PIN_H, PIN_W, 3))
    t[bar] = 0.0
    return t, bar


def test_streak_is_absent_when_there_is_no_pin():
    """Safe to leave on: a frame with nothing opaque gets nothing added."""
    assert F.specular_streak(np.ones((PIN_H, PIN_W, 3))).max() == 0.0


def test_streak_ignores_a_body_that_is_not_shank_shaped():
    """A compact blob has no long axis to speak of -- the two eigenvalues are
    nearly equal and the ridge angle would be whichever way the noise fell."""
    blob = np.ones((PIN_H, PIN_W, 3))
    blob[180:300, 300:420] = 0.0                         # 120 x 120
    assert F.specular_streak(blob).max() == 0.0


def test_streak_ignores_the_loop_fiber():
    """A 20 um fiber is under 3 px across.  The erosion is what keeps the
    glint on the pin and off the loop, so this is the guard on `min_width`."""
    yy, xx = np.mgrid[0:PIN_H, 0:PIN_W].astype(float)
    fiber = np.ones((PIN_H, PIN_W, 3))
    fiber[(np.abs(yy - 240) < 1.5) & (xx > 100)] = 0.0
    assert F.specular_streak(fiber).max() == 0.0


def test_streak_stays_inside_the_pin():
    """It may never touch the background: the ridge is masked by the tracer's
    own opacity, so a leak would mean the geometry escaped the body."""
    t, bar = _pin_frame()
    assert not F.specular_streak(t)[~bar].any()


def test_streak_matches_the_measured_cross_section():
    """Against A01 (f=0.262, FWHM 0.150 w) and E02 (f=0.288, FWHM 0.112 w).

    Grain off, because a single column of a grainy ridge is not a shape.
    """
    t, bar = _pin_frame()
    out = F.apply_camera(t, streak_params={"grain": 0.0})[:, 550, 0] * 255
    rows = np.nonzero(bar[:, 550])[0]
    floor = np.percentile(out[rows], 20)
    pk = rows[int(np.argmax(out[rows]))]
    assert 0.24 < (pk - rows[0]) / rows.size < 0.32          # ridge position
    half = floor + 0.5 * (out[pk] - floor)
    lo, hi = pk, pk
    while out[lo] > half:
        lo -= 1
    while out[hi] > half:
        hi += 1
    assert 0.10 < (hi - lo) / rows.size < 0.17               # ridge FWHM
    assert 1.6 < out[pk] / floor < 4.0                       # peak, x the floor


def test_streak_grain_is_deterministic():
    """Load-bearing.  `test_server_settle_parity` compares JPEG bytes between
    the live server and a fresh render; an RNG here would break that guard
    instead of this one."""
    t, _ = _pin_frame()
    assert np.array_equal(F.specular_streak(t), F.specular_streak(t.copy()))


def test_streak_grain_rides_with_the_pin():
    """Surface roughness belongs to the PIN, not to the camera -- unlike the
    illumination field, which is fixed in camera space.  Translate the pin and
    the grain must translate with it, or it crawls across the shank as the
    stage pans (and becomes a localisation shortcut for anything trained on
    these frames)."""
    def at(dx):
        yy, xx = np.mgrid[0:PIN_H, 0:PIN_W].astype(float)
        bar = ((np.abs((yy - 240) - 0.04 * (xx - 352 - dx)) < 47.5)
               & (np.abs(xx - 352 - dx) < 150))          # wholly inside frame
        t = np.ones((PIN_H, PIN_W, 3))
        t[bar] = 0.0
        return t

    a = F.specular_streak(at(0))
    b = F.specular_streak(at(30))
    assert a.max() > 0.1
    assert np.allclose(a[:, :-30], b[:, 30:], atol=1e-12)


def test_streak_does_not_put_the_white_rail_back_in_reach():
    """The whole point of the affine operator is that neither rail is
    reachable.  An additive term could undo that; this says it does not."""
    t, _ = _pin_frame()
    out = F.apply_camera(t)
    assert out.min() > 0.0 and out.max() < 1.0


def test_streak_follows_a_rotated_pin():
    """The axis comes from the image's own second moments, so the glint tracks
    the pin through any spindle angle without this stage seeing the pose."""
    yy, xx = np.mgrid[0:PIN_H, 0:PIN_W].astype(float)
    for tilt in (-0.6, 0.0, 0.6):
        bar = (np.abs((yy - 240) - tilt * (xx - 352)) < 47.5) & (np.abs(xx - 352) < 180)
        t = np.ones((PIN_H, PIN_W, 3))
        t[bar] = 0.0
        s = F.specular_streak(t)
        assert s.max() > 0.1, f"no streak at tilt {tilt}"
        assert not s[~bar].any(), f"streak leaked off the pin at tilt {tilt}"


def test_streak_refuses_a_body_whose_width_the_frame_cuts():
    """`mitegen_200um`'s 1 um pixels put a 0.7 mm pin wider than the frame, so
    its moment aspect wanders 1.1-2.2 with the spindle and a bare aspect
    threshold made the glint blink on and off six times a revolution.  A side
    against the frame means half_w -- which sets both the ridge's position and
    its FWHM -- was never measurable, so there is nothing to draw."""
    t = np.ones((PIN_H, PIN_W, 3))
    t[:, 100:600] = 0.0                          # runs off the top AND bottom
    assert F.specular_streak(t).max() == 0.0


def test_streak_tapers_the_tip_but_not_the_frame_edge():
    """An end the FRAME cut is not an end: the shank continues past it, and
    fading there put a fake taper on the last 8 px of every hampton frame."""
    yy, xx = np.mgrid[0:PIN_H, 0:PIN_W].astype(float)
    t = np.ones((PIN_H, PIN_W, 3))
    t[(np.abs(yy - 240) < 47.5) & (xx > 400)] = 0.0     # tip at 400, runs off right
    r, c, val = F._streak_patch(t, {"grain": 0.0})       # grain is not a shape
    peak = np.zeros(PIN_W)
    for cc in np.unique(c):
        peak[cc] = val[c == cc].max()
    full = peak[500:650].max()
    assert peak[PIN_W - 1] > 0.9 * full, "faded at the frame edge"
    assert peak[405] < 0.6 * full, "did not taper at the tip"


def test_streak_can_be_switched_off():
    t, _ = _pin_frame()
    on = F.apply_camera(t)
    off = F.apply_camera(t, streak=False)
    assert not np.array_equal(on, off)
    assert np.array_equal(off, F.apply_camera(t, streak_params={"gain": 0.0}))


def test_value_noise_has_unit_sd_and_the_measured_correlation_length():
    """`grain` reads as 'sd, x the ridge amplitude' only if the noise is
    normalised, and the reference frames put the correlation at 2 px."""
    u, v = np.mgrid[0:600, 0:600].astype(float)
    n = F._value_noise(u, v, F.STREAK["grain_px"], F.STREAK["seed"])
    assert n.std() == pytest.approx(1.0, abs=0.05)
    r = n - n.mean()
    c0 = float((r * r).mean())
    lag = next(k for k in range(1, 10)
               if float((r[:, :-k] * r[:, k:]).mean()) / c0 < 1 / np.e)
    assert 2 <= lag <= 4


def test_wide_opaque_is_a_box_erosion():
    """The doubling shift-and must agree with the obvious implementation --
    it is 10x faster and that is the only reason it is written that way."""
    rng = np.random.default_rng(7)
    m = rng.random((80, 90)) < 0.6
    m[20:60, 30:70] = True
    k = 9
    got = F._wide_opaque(m, k)
    want = np.zeros_like(m)
    for i in range(k // 2, m.shape[0] - k // 2):
        for j in range(k // 2, m.shape[1] - k // 2):
            want[i, j] = m[i - k//2:i + k//2 + 1, j - k//2:j + k//2 + 1].all()
    assert np.array_equal(got, want)


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
