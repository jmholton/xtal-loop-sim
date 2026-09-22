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
SCENE_DIR = os.path.join(REPO_ROOT, "data", "scene_files")
LIB_ROOT = os.path.join(REPO_ROOT, "data", "frame_library")

from loop_sim.renderer import field as F

H, W = 96, 128


def _t(value):
    return np.full((H, W, 3), float(value))


def _composed(man, lib_dir, rec, box, out_size):
    """The pose's crop, composed the way the server composes it.

    NOT `Image.open(f).resize(out, box=box)`.  Templates store a tight crop of
    the rendered window, so a virtual `box` applied straight to the file samples
    the wrong region -- and PIL pads rather than raising, so it would fail as a
    plausible-looking picture rather than an error.  Going through
    `TemplateSource` also means these tests cannot drift from the delivery path
    they exist to check.
    """
    from loop_sim.server.camera_server import TemplateSource
    return TemplateSource(man, lib_dir)._compose(rec, box, out_size)


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


# --- determinism: test_server_settle_parity needs exact bytes here ---------

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


def test_field_carries_the_mottle_the_quadratic_discards():
    """A 6-coefficient bowl explains 83.5% of the real field and the rest is
    soft blotchiness -- fitting the smooth part and throwing away the residual
    is what made an early render read as a flat grey card beside a photograph.
    Real frames leave sd 2.6% (C07) and 2.9% (A01) of level after a per-frame
    quadratic is removed.
    """
    H, W = 480, 704
    f = F.vignette(H, W)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    u, v = 2 * xx / (W - 1) - 1, 2 * yy / (H - 1) - 1
    A = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], -1).reshape(-1, 6)
    coef, *_ = np.linalg.lstsq(A, f.ravel(), rcond=None)
    resid = (f.ravel() - A @ coef).std() / f.mean()
    assert 0.03 < resid < 0.045, f"mottle is {100*resid:.1f}% of level, real is 3.4-3.9%"

    # Not "the mottled field varies more" -- it is a MULTIPLICATIVE term, so it
    # correlates with the bowl and can lower the total spread while adding
    # structure.  What must be true is that it adds structure the quadratic
    # cannot absorb, and that switching it off removes it.
    flat = F.vignette(H, W, mottle=0.0)
    cf, *_ = np.linalg.lstsq(A, flat.ravel(), rcond=None)
    assert (flat.ravel() - A @ cf).std() / flat.mean() < 1e-6
    assert not np.allclose(f, flat)


def test_mottle_has_energy_at_the_scales_the_real_frames_do():
    """Real background energy does not sit at one scale -- box-blurring the
    residual leaves 3.0-3.8% at 16-64 px and 4.6-7.0% at 128-256 px.  A single
    cell size read as a smooth wash and was invisible; six octaves is what
    puts cloud at more than one size."""
    f = F.vignette(480, 704)
    f = f / f.mean()
    fine = f - _boxblur(f, 33)               # detail finer than ~33 px
    coarse = _boxblur(f, 129)                # structure coarser than ~129 px
    assert fine.std() > 0.010, "no fine-scale structure"
    assert (coarse - coarse.mean()).std() > 0.015, "no coarse-scale structure"


def _boxblur(a, k):
    ker = np.ones(k) / k
    pad = k // 2
    out = np.apply_along_axis(
        lambda r: np.convolve(np.pad(r, pad, mode="reflect"), ker, "valid"), 1, a)
    return np.apply_along_axis(
        lambda c: np.convolve(np.pad(c, pad, mode="reflect"), ker, "valid"), 0, out)


def test_mottle_moves_the_field_shape_but_never_its_level():
    """`E0` is a measured number. The blobs perturb about it; they may not
    shift it, or every served frame's exposure drifts."""
    for h, w in ((480, 704), (96, 128), (240, 320)):
        flat = F.vignette(h, w, mottle=0.0)
        mott = F.vignette(h, w)
        assert mott.mean() == pytest.approx(flat.mean(), rel=1e-9)


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
    """`test_server_settle_parity` compares JPEG bytes between the live
    server and a fresh render, so this must be bit-identical."""
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
    """A transmittance frame holding one opaque bar, and the `pin` for it.

    Returns `(t, bar, pin)`.  The geometry is derived from the SAME numbers the
    bar is drawn from, so a test can never accidentally assert that a fit found
    the bar -- there is no fit any more.  `pin` is what
    `pin_projection.project_pin` would produce for this bar: the perpendicular
    half-width is the vertical one foreshortened by the tilt, and the tip at
    `x_from` is a real end while the right-hand end is one the frame cut.
    """
    yy, xx = np.mgrid[0:PIN_H, 0:PIN_W].astype(float)
    bar = (np.abs((yy - 240) - tilt * (xx - 350)) < width / 2.0) & (xx > x_from)
    t = np.ones((PIN_H, PIN_W, 3))
    t[bar] = 0.0
    norm = (1.0 + tilt * tilt) ** 0.5
    x0 = 0.5 * (x_from + PIN_W - 1)
    pin = (x0, 240.0 + tilt * (x0 - 350.0), 1.0 / norm, tilt / norm,
           0.5 * width / norm, 0.5 * (PIN_W - 1 - x_from) * norm, False, True)
    return t, bar, pin


def test_streak_is_absent_when_there_is_no_pin():
    """Safe to leave on: no pin projected means nothing is added.

    `pin=None` is the answer for a pose that has panned or zoomed past the pin,
    for a mount seen end-on, and for a scene with no shiny body at all.  It is
    also the DEFAULT, so a caller that does not know where the pin is gets no
    glint rather than a guessed one -- a missing glint is a small
    incorrectness, a glint on the wrong body is a false feature.
    """
    t, _, pin = _pin_frame()
    assert F.specular_streak(np.ones((PIN_H, PIN_W, 3)), None).max() == 0.0
    assert F.specular_streak(t, None).max() == 0.0
    assert np.array_equal(F.apply_camera(t), F.apply_camera(t, streak=False))


def test_streak_is_absent_when_the_only_dark_body_is_not_the_pin():
    """The glint's geometry comes from the scene, not from thresholding
    dark pixels, so a frame whose pin is out of view gets nothing no
    matter how large or well-isolated a dark region is.  See
    docs/DECISIONS.md 2026-08-12 (glint projected from scene).
    """
    blob = np.ones((PIN_H, PIN_W, 3))
    blob[150:330, 240:480] = 0.0                         # 240 x 180, well eroded
    assert F.specular_streak(blob, None).max() == 0.0


def test_streak_stays_inside_the_pin():
    """It may never touch the background: the ridge is masked by the tracer's
    own opacity AND by the projected shank, so a leak would mean the geometry
    escaped the body."""
    t, bar, pin = _pin_frame()
    assert not F.specular_streak(t, pin)[~bar].any()


def test_streak_stays_off_a_second_dark_body():
    """A second dark body on the ridge's own line must get no glint: the
    band is bounded by the projected shank, not a frame-wide mask over
    everything dark.  See docs/DECISIONS.md 2026-08-12 (glint projected
    from scene).
    """
    t, bar, pin = _pin_frame()
    intruder = np.zeros_like(bar)
    intruder[200:280, 40:240] = True        # dark, wide, ON the ridge's line
    t[intruder] = 0.0
    s = F.specular_streak(t, pin)
    assert not s[intruder].any(), "the glint leaked onto a second body"
    assert s[bar].max() > 0.1, "and it must still draw on the pin"


def test_streak_matches_the_measured_cross_section():
    """Against A01 (f=0.262, FWHM 0.150 w) and E02 (f=0.288, FWHM 0.112 w).

    Grain off, because a single column of a grainy ridge is not a shape.
    """
    t, bar, pin = _pin_frame()
    out = F.apply_camera(t, streak_params={"grain": 0.0}, pin=pin)[:, 550, 0] * 255
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
    """`test_server_settle_parity` compares JPEG bytes between the live
    server and a fresh render; an RNG here would break that guard instead
    of this one."""
    t, _, pin = _pin_frame()
    assert np.array_equal(F.specular_streak(t, pin),
                          F.specular_streak(t.copy(), pin))


def test_streak_grain_scintillates_with_the_pose_but_holds_at_rest():
    """A machined shank is rough at the wavelength scale, so as it turns,
    different micro-facets enter the specular condition and the glint TWINKLES.

    An earlier version hashed the grain on the pin's own frame so it would ride
    with it.  That is right for a static surface texture and wrong for this:
    it also slid against the pin whenever the visible portion changed, which
    read as parallax and gave the effect away as painted on.

    Both halves matter.  Re-rolling on pose change is the effect; holding still
    at a FIXED pose is what keeps the whole chain a pure function of what is
    being rendered, which `test_server_settle_parity` compares bytes against.
    """
    t, _, pin = _pin_frame()
    a = F.specular_streak(t, pin, {"phase": 1234})
    again = F.specular_streak(t, pin, {"phase": 1234})
    moved = F.specular_streak(t, pin, {"phase": 5678})

    assert np.array_equal(a, again), "a held pose must not shimmer"
    assert not np.array_equal(a, moved), "the pattern must re-roll on a new pose"
    lit = a > 0
    assert np.abs(a - moved)[lit].mean() > 1e-3          # visibly different
    # ...but only the grain moves: the ridge itself is in the same place.
    assert np.array_equal(lit, moved > 0)
    assert abs(a[lit].mean() - moved[lit].mean()) < 0.05 * a[lit].mean()


def test_pose_phase_is_stable_and_discriminating():
    """The phase must come from the POSE, so the same pose renders the same
    bytes and any real move re-rolls the grain."""
    from loop_sim.server.camera_server import pose_phase

    base = {"rotx": 37.0, "tx": 0.1, "zoom": 1.5}
    assert pose_phase(base) == pose_phase(dict(base))
    assert pose_phase(base) == pose_phase(dict(base, rotx=37.00000001))  # quantised
    assert pose_phase(base) != pose_phase(dict(base, rotx=37.01))
    assert pose_phase(base) != pose_phase(dict(base, tx=0.1005))
    assert pose_phase({}) == pose_phase({"rotx": 0.0})


def test_streak_does_not_put_the_white_rail_back_in_reach():
    """The whole point of the affine operator is that neither rail is
    reachable.  An additive term could undo that; this says it does not."""
    t, _, pin = _pin_frame()
    out = F.apply_camera(t, pin=pin)
    assert out.min() > 0.0 and out.max() < 1.0


def test_streak_follows_a_rotated_pin():
    """The axis is projected from the scene, so the glint tracks the pin
    through any spindle angle, and it never wanders off the body."""
    for tilt in (-0.6, 0.0, 0.6):
        t, bar, pin = _pin_frame(width=95, tilt=tilt, x_from=172)
        s = F.specular_streak(t, pin)
        assert s.max() > 0.1, f"no streak at tilt {tilt}"
        assert not s[~bar].any(), f"streak leaked off the pin at tilt {tilt}"


def test_streak_refuses_mitegen_at_every_angle():
    """`mitegen_200um`'s mount is not a shank the glint models, and it has to
    be refused at EVERY angle, not most: a glint that blinks on and off six
    times a revolution is far worse than one that never appears, and that is
    exactly what an earlier aspect-threshold version did.

    Now settled from the scene rather than guessed from the picture.  Its pin
    is `axis [0,0,1]` -- the BEAM axis -- so at phi=0 it is end-on with no
    shank in view, and at every other angle its 0.5 mm diameter is 500 px
    against a 480-row frame at 1 um pixels, which is wider than the frame's
    short side and leaves the ridge's position undefined.  Both refusals are
    computed, not measured off a silhouette.

    Run against the shipped library rather than a synthetic, because the
    synthetic that replaced it was a clean bar running off two edges -- a body
    the old fit should and did accept -- so it tested the opposite of the bug.
    """
    import json
    from PIL import Image
    from loop_sim.library.frame_library import frame_for_angle, pose_crop
    from loop_sim.motors.goniometer import Goniometer
    from loop_sim.renderer.pin_projection import project_pin, template_mapper
    from loop_sim.scene.scene import load as load_scene

    lib = os.path.join(LIB_ROOT, "mitegen_200um")
    man_path = os.path.join(lib, "manifest.json")
    if not os.path.exists(man_path):
        pytest.skip("mitegen_200um library not present")
    with open(man_path) as fh:
        man = json.load(fh)
    scene = load_scene(os.path.join(SCENE_DIR, "mitegen_200um.yaml"))

    fired = []
    for ang in range(0, 360, 15):
        rec = frame_for_angle(man, float(ang))
        box, out, _, _ = pose_crop(man, angle_deg=float(ang), clamp=True)
        crop = _composed(man, lib, rec, box, out)
        t = F.to_sensor(np.asarray(crop, np.float64) / 255.0)
        to_px, frame_wh = template_mapper(man, box, out, F.SENSOR_WH)
        gono = Goniometer(scene.geometry).set(**{man["axis"]: float(ang)})
        pin = project_pin(scene, gono, to_px, frame_wh)
        if F._streak_patch(t, pin) is not None:
            fired.append(ang)
    assert not fired, f"glint drawn on mitegen at {fired}"


def test_streak_never_lands_on_the_droplet():
    """The regression this test guards, on the real library.

    On `hampton_300um_realistic` the pin's metal starts at lab x = 1.000 mm
    and the loop, stem and droplet all live below x = 0.9; not one streak
    pixel may fall there.  See docs/DECISIONS.md 2026-08-12 (glint
    projected from scene).

    Driven through the real delivery chain (`pose_crop`, `to_sensor`,
    `project_pin`, `_streak_patch`): every defect this glint has had was
    found by driving it, none by a synthetic frame.
    """
    import json
    from PIL import Image, ImageFilter
    from loop_sim.library.frame_library import frame_for_angle, pose_crop
    from loop_sim.motors.goniometer import Goniometer
    from loop_sim.renderer.pin_projection import project_pin, template_mapper
    from loop_sim.scene.scene import load as load_scene

    lib = os.path.join(LIB_ROOT, "hampton_300um_realistic")
    man_path = os.path.join(lib, "manifest.json")
    if not os.path.exists(man_path):
        pytest.skip("hampton_300um_realistic library not present")
    with open(man_path) as fh:
        man = json.load(fh)
    scene = load_scene(os.path.join(SCENE_DIR, "hampton_300um_realistic.yaml"))
    win, tw = man["window_mm"], int(man["rendered"]["width"])
    sensor_w = F.SENSOR_WH[0]

    leaks, drawn = [], 0
    for zoom in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        for ang in (0, 15, 30, 45, 90, 135, 180, 270):
            rec = frame_for_angle(man, float(ang))
            box, out, sigma, _ = pose_crop(man, angle_deg=float(ang),
                                           zoom=zoom, clamp=True)
            crop = _composed(man, lib, rec, box, out)
            if sigma > 0.05:
                crop = crop.filter(ImageFilter.GaussianBlur(radius=sigma))
            t = F.to_sensor(np.asarray(crop, np.float64) / 255.0)
            to_px, frame_wh = template_mapper(man, box, out, F.SENSOR_WH)
            gono = Goniometer(scene.geometry).set(**{man["axis"]: float(ang)})
            patch = F._streak_patch(t, project_pin(scene, gono, to_px, frame_wh),
                                    {"phase": 0}, sigma)
            if patch is None:
                continue
            drawn += 1
            # column -> lab x, through the crop box and the sensor resample
            src = box[0] + (patch[1] / sensor_w) * (box[2] - box[0])
            lab_x = win["x0"] + src * (win["x1"] - win["x0"]) / tw
            off = int((lab_x < 0.9).sum())
            if off:
                leaks.append((zoom, ang, off, patch[0].size))
    assert not leaks, f"glint drew on the loop/droplet at {leaks[:6]}"
    assert drawn >= 8, "the glint must still be drawn where the pin IS in view"


def test_projected_pin_matches_the_rendered_silhouette():
    """Architecture-independent check: the projection must agree with the
    picture, where the picture is unambiguous.

    On `hampton_300um_realistic` at zoom 1 the pin is the only wide dark body
    in the right half of the frame, so its band can be measured straight off a
    column.  The projected centre row and half-width must match it.  This is
    the class of check DATA.md's "Known gaps" asks for -- it compares against
    the render rather than against another computation on the same box, so a
    wrong transform cannot pass it.
    """
    import json
    from PIL import Image
    from loop_sim.library.frame_library import frame_for_angle, pose_crop
    from loop_sim.motors.goniometer import Goniometer
    from loop_sim.renderer.pin_projection import project_pin, template_mapper
    from loop_sim.scene.scene import load as load_scene

    lib = os.path.join(LIB_ROOT, "hampton_300um_realistic")
    man_path = os.path.join(lib, "manifest.json")
    if not os.path.exists(man_path):
        pytest.skip("hampton_300um_realistic library not present")
    with open(man_path) as fh:
        man = json.load(fh)
    scene = load_scene(os.path.join(SCENE_DIR, "hampton_300um_realistic.yaml"))

    for zoom, col in ((1.0, 660), (1.5, 660), (2.0, 690)):
        rec = frame_for_angle(man, 0.0)
        box, out, _, _ = pose_crop(man, angle_deg=0.0, zoom=zoom, clamp=True)
        crop = _composed(man, lib, rec, box, out)
        t = F.to_sensor(np.asarray(crop, np.float64) / 255.0)
        dark = np.nonzero(t[:, col, 0] <= F.STREAK["opaque"])[0]
        assert dark.size > 20, f"no pin band at zoom {zoom}, column {col}"

        to_px, frame_wh = template_mapper(man, box, out, F.SENSOR_WH)
        gono = Goniometer(scene.geometry).set(rotx=0.0)
        pin = project_pin(scene, gono, to_px, frame_wh)
        assert pin is not None, f"pin not projected at zoom {zoom}"
        _, y0, _, _, half_w, _, _, _ = pin

        assert abs(y0 - 0.5 * (dark[0] + dark[-1])) < 2.0, (
            f"zoom {zoom}: projected row {y0:.1f} vs measured "
            f"{0.5 * (dark[0] + dark[-1]):.1f}")
        assert abs(half_w - 0.5 * dark.size) < 3.0, (
            f"zoom {zoom}: projected half-width {half_w:.1f} vs measured "
            f"{0.5 * dark.size:.1f}")


def test_streak_tapers_the_tip_but_not_the_frame_edge():
    """An end the FRAME cut is not an end: the shank continues past it, and
    fading there put a fake taper on the last 8 px of every hampton frame."""
    t, _, pin = _pin_frame(width=95, tilt=0.0, x_from=400)
    r, c, val = F._streak_patch(t, pin, {"grain": 0.0})   # grain is not a shape
    peak = np.zeros(PIN_W)
    for cc in np.unique(c):
        peak[cc] = val[c == cc].max()
    full = peak[500:650].max()
    assert peak[PIN_W - 1] > 0.9 * full, "faded at the frame edge"
    assert peak[405] < 0.6 * full, "did not taper at the tip"


def test_streak_defocuses_with_the_sample():
    """The glint is light off the pin's SURFACE, so when the sample goes out of
    focus the glint must go with it -- it cannot stay razor-sharp on a pin that
    has visibly softened.  `pose_crop` already blurs the silhouette by this
    sigma; the glint now gets the same one.
    """
    def grain_of(a):
        """Residual along the ridge after its slow along-axis trend is removed.

        NOT the spread over all lit pixels -- that is dominated by the ridge's
        own Gaussian cross-section, which barely moves under a small blur, and
        measuring it that way hides the effect entirely.
        """
        row = int(np.argmax(a.max(axis=1)))
        line = a[row, :]
        line = line[line > 0]
        if line.size < 40:
            return 0.0
        k = np.ones(9) / 9.0
        smooth = np.convolve(np.pad(line, 4, mode="edge"), k, "valid")
        return float((line - smooth).std())

    t, _, pin = _pin_frame()
    sharp = F.specular_streak(t, pin)
    g_sharp = grain_of(sharp)
    assert g_sharp > 0, "no grain to soften"
    for sigma, expect in ((1.5, 0.6), (4.0, 0.2)):
        soft = F.specular_streak(t, pin, defocus=sigma)
        g_soft = grain_of(soft)
        assert g_soft < expect * g_sharp, (
            f"sigma {sigma}: grain {g_soft:.5f} vs sharp {g_sharp:.5f}")
        # energy is spread, not destroyed, and it reaches further across the pin
        assert soft.sum() == pytest.approx(sharp.sum(), rel=0.15)
        assert (soft > 0).sum() > (sharp > 0).sum()
    assert np.array_equal(F.specular_streak(t, pin, defocus=0.0), sharp)


def test_defocus_blur_is_deterministic_and_conserves_energy():
    """It sits inside the byte-compared delivery chain, so it must be a pure
    function; and a normalised kernel must not change the total."""
    a = np.zeros((80, 90))
    a[40, 45] = 1.0
    b1, b2 = F._blur2d(a, 2.5), F._blur2d(a.copy(), 2.5)
    assert np.array_equal(b1, b2)
    assert b1.sum() == pytest.approx(1.0, rel=1e-9)
    assert b1[40, 45] < 0.1                              # actually spread
    assert F._blur2d(a, 5.0)[40, 45] < b1[40, 45]        # wider sigma, flatter


def test_streak_can_be_switched_off():
    t, _, pin = _pin_frame()
    on = F.apply_camera(t, pin=pin)
    off = F.apply_camera(t, streak=False, pin=pin)
    assert not np.array_equal(on, off)
    assert np.array_equal(off, F.apply_camera(t, streak_params={"gain": 0.0},
                                              pin=pin))


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
