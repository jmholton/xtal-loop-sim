"""Camera emulation: sensor raster, illumination field, black floor, tone.

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

It also supplies the sensor's RASTER (`to_sensor`, `SENSOR_WH`).  The tracer
renders square pixels; the BL831 camera's are 1.110 non-square and it emits
704x480.  That is a resample, not a rescale -- the field of view is the same
either way, to under 1% -- and the constant above says why 640 was right all
along and what the resample buys.

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

# The real sensor's raster.  Every frame in `real_images/` is 704x480, and the
# BL831 sample camera's pixels are NOT square: 6.7324 x 7.4729 um at the mid
# zoom stop and 0.8233 x 0.9139 at the hi stop -- aspect 1.1100 at both.  The
# renderer has one scalar `pixel_size`, so the shipped scenes model that camera
# the only way a square-pixel tracer can: 640 x 7.4 um covers 4736.0 um where
# 704 x 6.7324 covers 4739.6, agreeing to 0.08% horizontally and 0.98%
# vertically.  640 is therefore not a discrepancy against the photographs, it
# is the square-pixel rendition of them (704/640 = 1.100 cancels the 1.110
# pixel aspect), and rendering 704 wide at 7.4 um would over-cover the field by
# +9.92%.  What 640 does NOT reproduce is the frame SHAPE, and consumers care:
# dcss stores a um-per-pixel constant for this camera, so a stand-in that emits
# 640 columns reads 10% wide horizontally.  Resampling to the sensor raster
# here closes that without touching the scene, the renderer or any template.
SENSOR_WH = (704, 480)

_cache = {}
_weight_cache = {}


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


def _axis_weights(n_src, n_dst):
    """Bilinear source indices and weights for one resampled axis.

    Pixel CENTRES, not edges: output centre i sits at source coordinate
    `(i + 0.5) * n_src / n_dst - 0.5`, which is PIL's and OpenCV's convention
    and the one `pose_crop` already assumes when it offsets the crop box by
    half a source pixel.  Aligning corners instead would shift the image by
    half an output pixel at this scale factor.
    """
    key = (n_src, n_dst)
    hit = _weight_cache.get(key)
    if hit is not None:
        return hit
    c = (np.arange(n_dst, dtype=np.float64) + 0.5) * (n_src / float(n_dst)) - 0.5
    c = np.clip(c, 0.0, n_src - 1.0)
    i0 = np.floor(c).astype(np.intp)
    i1 = np.minimum(i0 + 1, n_src - 1)
    hit = (i0, i1, c - i0)
    _weight_cache[key] = hit
    return hit


def to_sensor(img, size=SENSOR_WH):
    """Resample a square-pixel render onto the camera's non-square raster.

    `img` is (H, W, 3) float; the result is (size[1], size[0], 3).  Returns the
    input unchanged when it is already that size, so this is free for a caller
    that renders at the sensor raster directly.

    WHY THIS RUNS AFTER THE DEFOCUS BLUR, NOT BEFORE.  The objective's PSF is
    isotropic in the optical image; it is the SENSOR that samples that image at
    two different pitches.  Blurring in square-pixel space with one scalar
    sigma and resampling afterwards reproduces that -- the blur comes out 1.1x
    wider vertically than horizontally in the delivered frame, which is what
    the real camera does.  Blurring after the resample would need an
    anisotropic kernel, and `ImageFilter.GaussianBlur` has no such thing.

    WHY BILINEAR.  A real sensor box-integrates over each pixel, but this is a
    1.100x UPSAMPLE: each output column draws on 0.909 source columns, so a box
    filter degenerates to nearly a point sample and bilinear is both the
    standard choice and the one the template crop upstream already uses.
    Separable and index-based, so it is a pure function of (shape, size) --
    `tests/test_server_settle_parity.py` compares JPEG bytes and would catch
    any per-call state here.
    """
    a = np.asarray(img, dtype=np.float64)
    h, w = a.shape[:2]
    tw, th = int(size[0]), int(size[1])
    if (w, h) == (tw, th):
        return a
    if w != tw:
        i0, i1, t = _axis_weights(w, tw)
        a = a[:, i0] * (1.0 - t)[None, :, None] + a[:, i1] * t[None, :, None]
    if h != th:
        j0, j1, t = _axis_weights(h, th)
        a = a[j0] * (1.0 - t)[:, None, None] + a[j1] * t[:, None, None]
    return a


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
