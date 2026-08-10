"""Camera emulation: illumination field, black floor, and tone response.

The ray tracer computes TRANSMITTANCE.  Every ray is born carrying radiance
exactly 1.0 (`microscope.py`, `engine_torch.py`) and a ray that hits nothing is
never multiplied by anything, so an empty field of view comes out at exactly
1.0 and an opaque object at exactly 0.0.  That is correct physics for what the
tracer models, and it is not a photograph.  Measured on the delivered hampton
frame: **84.6% of pixels exactly 255, 14.4% exactly 0, 1.1% anything else**.
Real BL831 sample-camera frames carry **19-27% genuinely intermediate tone**
(`real_images/C07` 27.4%, `D03` 19.0%) with an empty field at ~0.65 of full
scale and an opaque pin at ~0.18 -- neither rail is ever reached.  So the
render is a binary silhouette where the photograph is continuous-tone, and that
-- not the background pattern -- is the largest single difference between them.

This module supplies the two terms the transport does not model:

    observed = (E(x,y) - B) * T(x,y) + B

`E` is the incident illumination field (what the camera sees at T=1) and `B` is
the black floor (what it sees at T=0: veiling glare in the objective plus the
sensor pedestal).  It is a lerp between the only two anchors that were actually
measured, which makes both rails unreachable BY CONSTRUCTION -- the real
frames' "40-226, nothing clipped" comes out for free, with no clamp and no tone
curve.  Nothing here is fitted to the space between those anchors, because
nothing was measured there.

WHY THIS IS NOT INSIDE EITHER TRACER, AND MUST NOT BE MOVED THERE.  Three
independent reasons, any one sufficient:

  * The illumination defect is fixed in CAMERA space.  Correlating the
    low-frequency background across one session -- four spindle angles and a
    sample translation -- gives r = 0.93-1.00: the sample moves, the pattern
    does not.  Applied after `pose_crop`, this stage is structurally incapable
    of panning with the sample.  Baked into a template it would rotate with it.
  * `frame_library.content_window` separates sample from background with
    |img - bg| > 2/255.  Any field upstream of it reads as content everywhere,
    the scout window widens to the 64x cap and the build guard refuses.
  * Templates keep storing raw transmittance, so the shipped frame libraries
    stay valid and nothing rebuilds for a change that never enters them.

DETERMINISM IS LOAD-BEARING.  `tests/test_server_settle_parity.py` compares
JPEG BYTES between the live server and a fresh render, and
`tests/test_torch_render_parity.py` asserts the two engines agree byte-exactly.
Everything here is a pure function of (shape, parameters) -- no RNG, no clock,
no per-call state.  If a future term needs grain (a rough pin scatters light
unevenly), derive it from pixel coordinates or a stored tile, never from
`np.random` without a fixed seed threaded through the call.

CALIBRATION PROVENANCE.  `E0`, the vignette shape and `B` were fitted to an
empty field reconstructed from 300 frames sampled across
`sfd/loop_centering_stuff/run4` (3,024 frames of one session, 2020-01-29), by
taking the per-pixel 80th percentile -- not the median, because the mount
occupies the left of frame in more than half the frames there and would
contaminate it.  The quadratic explains **83.5%** of the field's variance
(residual sd 6.63/255, 4.4% of level).

A CAVEAT WORTH READING BEFORE TRUSTING THE DEFAULT AMPLITUDE.  There is no
single true field.  The 2020 pattern correlates r = 0.99-1.00 with its own
session but only 0.11 / -0.31 / -0.18 / +0.35 with the 2005 / 2021 / 2025 /
2026 epochs, and its amplitude (sd/median 0.090) is 5-7x LARGER than every
other epoch measured (0.013-0.023) -- including the two most recent production
frames.  The shipped default reproduces the reference the work was judged
against; it is a plausible field, not a universal one.  `amplitude` is a
scalar for exactly this reason, and a future caller that wants per-frame
variation (a model trained on a background fixed in camera space can learn to
localise by it instead of by the loop) should vary the coefficients rather than
add noise here.
"""
import numpy as np

# Empty-field level as a fraction of full scale.  Measured 152.2/255 on the
# reconstructed field; real single frames of that session read 164-169 in their
# brighter regions, which the vignette shape accounts for.
E0 = 0.5969

# Black floor: what an opaque object reads.  Measured as the 5th percentile of
# sub-90 pixels over the same 300 frames = 45/255, which lands on the 44-48
# pin interior measured independently in C07.  Veiling glare in the objective
# and the sensor pedestal are not separable from these frames -- the background
# only varies 8% across a frame, so a constant-ratio and a constant-offset model
# fit about equally well (CV 10.2% vs 7.8%).  The affine form takes the offset
# reading; a blur-based glare term is the documented next refinement.
B = 0.1765

# Vignette shape on normalised coordinates u, v in [-1, 1], in the basis
# [1, u, v, u^2, u*v, v^2], scaled so the field's mean is 1.0.  The dominant
# term is v^2 at -0.304: a vertical bowl, bright through the middle rows and
# falling off top and bottom, peak-to-trough 41.6% of level.  A plane alone
# explains under 1% of this -- the linear terms very nearly cancel by symmetry,
# which is why a gradient model looks like it fails.
VIGNETTE = (1.09472, -0.02534, +0.05636, +0.02100, -0.00112, -0.30397)

# Rec. 601 luma weights, used only by the `mono` option.
_LUMA = np.array([0.299, 0.587, 0.114])

_cache = {}


def vignette(h, w, coeffs=VIGNETTE, amplitude=1.0):
    """The illumination field shape, (h, w) float64 with mean ~1.0.

    Coordinates are normalised to [-1, 1] on each axis, so the same
    coefficients describe the same physical field at any render size --
    important because templates are built at one resolution and served at
    another.  `amplitude` scales the departure from flat: 0.0 is a uniform
    field, 1.0 is the measured 2020 session, and values below 1 match the
    flatter recent epochs (see the module docstring).

    Note this indexes OUTPUT pixels and is not rescaled by zoom, which asserts
    the defect sits downstream of the zoom optics.  That is an assumption, not
    a measurement; if the mottle is later found to scale with zoom it sits near
    a field-conjugate plane instead.
    """
    key = (h, w, coeffs, float(amplitude))
    hit = _cache.get(key)
    if hit is not None:
        return hit
    v, u = np.mgrid[0:h, 0:w].astype(np.float64)
    u = 2.0 * u / max(w - 1, 1) - 1.0
    v = 2.0 * v / max(h - 1, 1) - 1.0
    c = coeffs
    f = (c[0] + c[1] * u + c[2] * v
         + c[3] * u * u + c[4] * u * v + c[5] * v * v)
    if amplitude != 1.0:
        f = 1.0 + amplitude * (f - 1.0)
    _cache[key] = f
    return f


def to_luma(img):
    """Collapse an (H, W, 3) image to grey, keeping three channels.

    Material colour in this renderer is an ABSORPTION spectrum
    (`mu_per_ch = mu_optical + 30*(1 - colour)`, `microscope.py`), so a scene
    that names a crystal `[0.7, 0.9, 1.0]` renders it strongly blue -- it
    absorbs red at 9/mm.  Real frames are near-neutral.  This masks that in
    camera space; it does not fix it.  The repair is `colour: [1, 1, 1]` with
    the absorption moved into `mu_optical`, which is a scene change and costs a
    library rebuild -- hence the option, and hence it should be switched off
    the day the YAML is right.
    """
    return np.repeat((img @ _LUMA)[..., None], 3, axis=2)


def apply_camera(img, level=E0, floor=B, coeffs=VIGNETTE, amplitude=1.0,
                 mono=False, enabled=True):
    """Map transmittance to what the camera would record.  Returns a new array.

    `img` is (H, W, 3) float in [0, 1] straight out of the tracer, or a crop of
    a template holding the same quantity.  The result is in [floor, level*max
    (shape)] and therefore never touches 0 or 255 after quantisation.
    """
    if not enabled:
        return img
    t = to_luma(img) if mono else img
    e = float(level) * vignette(t.shape[0], t.shape[1], coeffs, amplitude)
    return (e[..., None] - float(floor)) * t + float(floor)
