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

# What the quadratic LEAVES BEHIND, put back.  The bowl above explains 83.5% of
# the field; the rest is the soft blotchiness every real frame has, and fitting
# the smooth part while discarding the remainder is why the first version of
# this stage read as a flat grey card next to a photograph.
#
# Deterministic, and FIXED IN CAMERA SPACE like the bowl it corrects: this is an
# illumination defect, so the sample moves under it.  That is the opposite of
# the pin's grain, which belongs to the pin and re-rolls with the pose -- the
# two look similar in the code and are physically unrelated.
#
# SIX OCTAVES, because one scale cannot look like cloud.  Splitting the real
# residual by successive box-blurs shows energy at every scale, not one:
#
#   features surviving a blur of   16px   32px   64px  128px  256px
#     C07                          3.19%  3.03%  3.79%  5.71%  4.59%
#     A01                          3.24%  3.05%  3.44%  5.17%  7.01%
#     F04                          3.71%  3.45%  3.15%  2.99%  3.06%
#
# so the base octave is set at 0.8 u-units (280 px at the 704-wide raster) and
# halved five times to ~9 px, which brackets that range.  A single 200 px
# cell -- the first version -- read as a smooth wash and the owner could not
# see it at all.
#
# The TOTAL is 3.6% of level, from re-measuring the real frames with a proper
# background mask.  The first measurement said 2.6-2.9% because it took
# "background" to mean brighter than the 60th percentile, which clips the dark
# half of every cloud and biases the spread down; the mask now dilates the dark
# body instead and reads 3.38 / 3.48 / 3.92%.
MOTTLE = 0.0435        # raw fBm amplitude, SOLVED for MOTTLE_RESIDUAL below
MOTTLE_RESIDUAL = 0.036  # what it must LEAVE after a quadratic is re-fitted
MOTTLE_UV = 0.80        # coarsest octave, in u-units (u spans 2.0 across w)
MOTTLE_OCTAVES = 6
# 0.90, not the textbook 0.5.  Real background energy is nearly FLAT across
# scale (3.0-3.8% at 16-64 px against 4.6-7.0% at 128-256), so a fast roll-off
# puts everything in the coarse octaves and the frame reads as a smooth wash:
# at gain 0.62 the sub-33 px detail was 0.39% of level against the real ~3.2%,
# and six octaves at 0.90 lift it to 1.61% for the same 3.6% total.
MOTTLE_GAIN = 0.90      # amplitude ratio between successive octaves
MOTTLE_SEED = 0x30771E


def _fbm(u, v, cell, seed, octaves=MOTTLE_OCTAVES, gain=MOTTLE_GAIN):
    """Sum of halving-scale value noise, normalised to unit standard deviation.

    Octaves are statistically independent (each gets its own hash salt), so
    they add in QUADRATURE -- dividing by the linear sum would leave the result
    short of unit sd and make the amplitude constant lie.
    """
    total = 0.0
    amp, power = 1.0, 0.0
    for i in range(int(octaves)):
        total = total + amp * _value_noise(u, v, cell / (2 ** i),
                                           seed + i * 7919)
        power += amp * amp
        amp *= gain
    return total / (power ** 0.5)

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


def vignette(h, w, coeffs=VIGNETTE, amplitude=1.0, mottle=MOTTLE):
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
    key = (h, w, coeffs, float(amplitude), float(mottle))
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
    if mottle:
        n = _fbm(u, v * (h / float(w)), MOTTLE_UV, MOTTLE_SEED)
        # The mottle may change the field's SHAPE and never its LEVEL -- E0 is
        # a measured number and the blobs are a perturbation about it.  Both
        # steps are needed: subtracting the mean makes the perturbation
        # zero-mean, and rescaling removes what the bowl-blob covariance still
        # puts back.  Skipping the first shifted the served mean 6.7%; skipping
        # the second left 0.24%, and test_clear_path_reads_the_empty_field
        # catches both.
        g = f * (1.0 + amplitude * float(mottle) * (n - n.mean()))
        f = g * (f.mean() / g.mean())
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


# ---------------------------------------------------------------------------
# The pin's specular streak
# ---------------------------------------------------------------------------
# A 0.7 mm mounting pin is machined steel, and the tracer models it as a purely
# opaque body: every ray that meets it dies, so it renders as a flat silhouette
# at the black floor.  Real ones carry a bright, broken glint running the whole
# length of the shank -- the specular return off a rough cylinder -- and it is
# the largest remaining structural difference between a rendered pin and a
# photographed one.  Modelling it properly means a BRDF and a specular bounce
# in the tracer; this is the cosmetic stand-in, and it is deliberately in
# camera space so it costs no library rebuild.
#
# MEASURED on the two reference frames that show a pin clearly, both at the mid
# stop (`real_images/A01_nylonloop_pinleft_mid.jpg` and
# `E02_digitize_source_mid.jpg`), by taking the pin's edges per column and
# resampling each column onto a normalised cross-section:
#
#                              A01        E02
#   pin width                85 px      100 px
#   floor / background     46 / 167   60 / 254
#   ridge centre           f = 0.738  f = 0.288     <- opposite sides
#     as half-widths off    +0.48      -0.42            of the pin's axis
#   ridge FWHM             0.150 w    0.112 w
#   peak above the floor   0.82 x bg  0.15 x bg     <- the one wide spread
#   grain sd ON the ridge  6.6 lv     10.0 lv
#   grain sd on the body   0.50 lv    1.30 lv       <- ~10x quieter
#   grain correlation      2 px       2 px
#
# Three things that spread are recorded rather than averaged away.  (1) The
# ridge sits ~0.24 half-widths OFF the axis, but on opposite sides in the two
# frames -- which side is an illumination-geometry property, not a pin
# property, so `offset` is signed and the default follows E02 (the frame this
# repo's own loop geometry was digitized from).  (2) The peak spans 0.15-0.82
# of background; a duller pin and a brighter field give the low end.  The
# default is deliberately nearer the quiet end, because a blown-out glint on an
# otherwise flat pin reads worse than no glint at all.  (3) The grain lives on
# the SPECULAR term, not on the diffuse floor -- surface slope modulates what
# is reflected, not what is absorbed -- so it is applied multiplicatively to
# the ridge, which also keeps it from touching the pin body.
#
# NOT MODELLED: the bright rim both frames show at the pin's far edge (f ~ 0.95,
# A01 and E02 alike).  That is edge diffraction plus the cylinder's grazing
# return, it needs a second term, and it is much less visible than the streak.
STREAK = {
    "opaque":     0.04,   # transmittance at or below which a pixel is opaque
    "min_width":  13,     # px; narrower opaque bodies get no streak (the
                          # 20 um loop fiber is ~2.7 px, the pin ~95)
    "min_aspect": 1.8,    # and no streak on a body that is not shank-shaped
    "axis_gate":  0.90,   # fit the axis only across slices at least this
                          # fraction of the widest -- drops the tapering tip
    "max_width":  0.90,   # and nothing wider than this fraction of the frame's
                          # SHORT side is a pin whose sides we can see
    "offset":    -0.45,   # ridge centre, in HALF-widths from the pin's axis
                          # (A01 sits at +0.48, E02 at -0.42; sign is the
                          # illumination's, so the default follows E02)
    "width":      0.13,   # ridge FWHM, as a fraction of the pin's full width
    "gain":       0.35,   # peak above the floor, x the local illumination
    "grain":      0.15,   # grain sd, x the local ridge amplitude
    "grain_px":   2.0,    # grain correlation length, px
    "seed":       0x5EED,
    "phase":      0,      # re-rolls the grain; the server feeds it the pose
}


def _hash01(i, j, seed):
    """Deterministic pseudo-random floats in [-1, 1) from integer cells.

    A hash, not an RNG: no state, no seeding order, no per-call variation.
    `test_server_settle_parity` compares JPEG bytes between the live server and
    a fresh render, so a stochastic grain would break that guard rather than
    this one.  uint64 throughout because numpy wraps unsigned arithmetic
    silently, which is what a mixing function wants.
    """
    a = (i.astype(np.int64) + (1 << 20)).astype(np.uint64) * np.uint64(73856093)
    b = (j.astype(np.int64) + (1 << 20)).astype(np.uint64) * np.uint64(19349663)
    h = a ^ b ^ np.uint64(seed)
    h ^= h >> np.uint64(13)
    h *= np.uint64(1274126177)
    h ^= h >> np.uint64(16)
    return (h & np.uint64(0xFFFF)).astype(np.float64) / 32767.5 - 1.0


# Unit-sd normaliser for `_value_noise`: smoothstep-interpolated cell values
# drawn uniformly from [-1, 1) come out at sd 0.4325 (measured over 2.25e6
# samples at cell 2, 3 and 4 -- 0.4325/0.4291/0.4292, so it is a property of
# the interpolant, not of the cell size or the seed).  Dividing by it makes
# `grain` read directly as "sd, as a fraction of the ridge amplitude".
_NOISE_SD = 0.4325


def _value_noise(u, v, cell, seed):
    """Smooth deterministic noise on a rotated grid, unit standard deviation.

    `u`, `v` are coordinates in the PIN's own frame, not the image's, so the
    grain rides with the pin as the stage pans and the spindle turns -- surface
    roughness belongs to the pin, unlike the illumination field, which is fixed
    in camera space.  Cell values are hashed (never interpolated from a stored
    tile) and blended with a smoothstep, so a sub-pixel move slides the pattern
    continuously instead of making it jump.
    """
    p, q = u / cell, v / cell
    i, j = np.floor(p), np.floor(q)
    fp, fq = p - i, q - j
    wp = fp * fp * (3.0 - 2.0 * fp)
    wq = fq * fq * (3.0 - 2.0 * fq)
    n00 = _hash01(i, j, seed)
    n10 = _hash01(i + 1.0, j, seed)
    n01 = _hash01(i, j + 1.0, seed)
    n11 = _hash01(i + 1.0, j + 1.0, seed)
    top = n00 * (1.0 - wp) + n10 * wp
    bot = n01 * (1.0 - wp) + n11 * wp
    return (top * (1.0 - wq) + bot * wq) / _NOISE_SD


def _wide_opaque(mask, k):
    """Pixels whose whole k x k neighbourhood is opaque: erosion by a box.

    Doubling shift-and, so a k-wide erosion costs ceil(log2(k)) boolean ANDs
    per axis instead of k.  This runs on every served frame, and it is worth
    the two loops: 0.15 ms against 1.6 ms for the integral-image form and
    ~8 ms for a k x k minimum filter, byte-identical to both.

    It is what separates the pin from everything else in the frame -- a 20 um
    loop fiber is under 3 px across and never survives k = 13, so the streak
    cannot land on it.
    """
    h, w = mask.shape
    out = np.zeros_like(mask)
    if h <= k or w <= k:
        return out
    m = mask
    for axis in (0, 1):
        rem, step = k - 1, 1
        while rem > 0:
            d = min(step, rem)
            m = (m[:-d, :] & m[d:, :]) if axis == 0 else (m[:, :-d] & m[:, d:])
            rem -= d
            step *= 2
    r = k // 2
    out[r:r + m.shape[0], r:r + m.shape[1]] = m
    return out


def _fit_one_orientation(core, k, horizontal):
    """Width-gated centreline fit for ONE slicing direction, or None.

    Returns `(x0, y0, ax, ay, half_w, half_l, has_end, cut_lo, cut_hi)`.
    None means this orientation is not a consistent bar, for either of the two
    reasons that matter -- see `_pin_axis`.
    """
    m = core if horizontal else core.T
    n_cross, n_along = m.shape
    m0 = k // 2 + 1

    present = m.any(axis=0)
    if present.sum() < 8:
        return None
    first = np.argmax(m, axis=0).astype(np.float64)
    last = (n_cross - 1 - np.argmax(m[::-1], axis=0)).astype(np.float64)
    width = np.where(present, last - first + 1.0, 0.0)
    keep = present & (width >= float(STREAK["axis_gate"]) * width.max())
    if keep.sum() < 8:
        return None

    # A SIDE against the frame: half_w was never measurable, and it sets both
    # the ridge's position and its FWHM.  This is what rules mitegen out.
    if first[keep].min() <= m0 or last[keep].max() >= n_cross - 1 - m0:
        return None

    raw_w = float(width[keep].mean())
    span = np.nonzero(present)[0]
    cut_lo, cut_hi = bool(span[0] <= m0), bool(span[-1] >= n_along - 1 - m0)
    # A clipped END must be a CLEAN SEVER -- the body's cross-section right at
    # the frame equal to the width it has been holding.  Measured: hampton
    # reads 1.00-1.05 at every zoom, a tilted bar 1.01, while mitegen's only
    # side-free orientation reads 0.63 and is correctly refused.
    for cut, idx in ((cut_lo, span[0]), (cut_hi, span[-1])):
        if cut and not (0.75 <= float(m[:, idx].sum()) / raw_w <= 1.30):
            return None

    s = np.nonzero(keep)[0].astype(np.float64)
    centre = 0.5 * (first[keep] + last[keep])
    slope, intercept = np.polyfit(s, centre, 1)
    s_mid = 0.5 * (float(span[0]) + float(span[-1]))
    c_mid = slope * s_mid + intercept
    # Erosion by k took (k-1)/2 off every side; add it back to both extents.
    half_l = 0.5 * (float(span[-1]) - float(span[0])) + (k - 1) / 2.0
    half_w = 0.5 * raw_w + (k - 1) / 2.0
    norm = (1.0 + slope * slope) ** 0.5
    geom = ((s_mid, c_mid, 1.0 / norm, slope / norm) if horizontal
            else (c_mid, s_mid, slope / norm, 1.0 / norm))
    return geom + (half_w, half_l, cut_lo or cut_hi, cut_lo, cut_hi)


def _pin_axis(core, k):
    """`(x0, y0, ax, ay, half_w, half_l, axis_certain, cut_lo, cut_hi)`, or None.

    ACCEPTED LIMITATION, and the one worth reading before adding a sixth
    heuristic here.  Everything below infers the pin from its SILHOUETTE, so
    the glint is unavoidably a function of what is in frame.  Two consequences
    are known and were accepted deliberately (owner, 2026-08-10) rather than
    fixed: the glint disappears when the pin's SIDE leaves the frame (its width
    stops being measurable), and on a view showing only the bevelled tip the
    fit has no shank sides to lock onto and the ridge follows the tip's curve.

    The real fix is not a better inference -- it is to stop inferring.  The
    server knows the scene and the pose, `hampton_300um.yaml`'s pin is a
    cylinder with `axis [1,0,0], radius 0.35`, and `pose_crop` already does the
    lab-to-image mapping.  Handing this stage `(x0, y0, ax, ay, half_w)` in
    output pixels would make all of it exact at any zoom, any crop and any
    angle, and would DELETE `_wide_opaque`, `_fit_one_orientation`, this
    function, the sever test and the width cap -- ~150 lines of heuristic for
    ~40 of projection, and faster, since no mask or erosion would be needed.
    It would also settle mitegen properly: a kapton mount is declared not to
    shine, rather than guessed at from its shape.

    Fitted from the body's two long SIDES -- not from its second moments -- and
    only across slices at least `axis_gate` of the widest, which is what makes
    it immune to the shape of the tip.

    THE MOMENT FIT WAS A REAL BUG, TWICE OVER, and both were visible.

    (1) `hampton_300um`'s pin carries a 45-degree chisel, and as the spindle
    turns that bevel's projected silhouette sweeps up and down.  The filled
    mask's principal axis followed it: **-6.40 to +6.39 degrees, sinusoidal in
    phi**, tilting the specular ridge 54.5 px across the pin over a revolution.
    A horizontal pin lit from a fixed direction shows a horizontal glint at
    every angle.  Gating on width drops the tapering tip, so what is fitted is
    two parallel edges, and the answer is 0.00 degrees at every phi.

    (2) Past ~1.5x zoom the pin's in-frame piece is WIDER THAN IT IS LONG --
    211 x 59 px at 2.77x -- so the moments called it vertical and the fit
    measured the shank's width as its length.  half_w came out 17.8 px on a
    187 px pin and the glint vanished on zoom-in.

    So the orientation is not guessed from moments at all: BOTH are fitted and
    the consistent one wins.  An orientation is inconsistent if the body's
    SIDES run off the frame (its width is then unmeasurable) or if a clipped
    END is not a clean sever (the body does not simply continue past the
    crop).  Exactly one survives for a pin at any zoom; for `mitegen_200um`,
    whose 1 um pixels put a 0.7 mm pin wider than the frame, NEITHER does --
    which is the answer that stops the glint blinking six times a revolution.
    Moments break a tie only when the body touches no edge at all, where they
    are reliable and `min_aspect` is still there to catch a blob.
    """
    fits = [f for f in (_fit_one_orientation(core, k, True),
                        _fit_one_orientation(core, k, False)) if f is not None]
    if not fits:
        return None
    if len(fits) == 1:
        return fits[0]
    ended = [f for f in fits if f[6]]
    if len(ended) == 1:
        return ended[0]
    ys, xs = np.nonzero(core)                       # fully in frame: moments
    dy, dx = ys - ys.mean(), xs - xs.mean()
    return fits[0] if (dx * dx).mean() >= (dy * dy).mean() else fits[1]


def _streak_patch(t, params=None):
    """`(rows, cols, values)` for the pin's glint, or None if there is no pin.

    The sparse form.  `specular_streak` is the dense wrapper; `apply_camera`
    uses this one so a frame whose streak covers ~20k pixels does not pay a
    337k-pixel multiply and add to deliver it.  One implementation, two
    shapes -- the discipline the JPEG-byte guards need.
    """
    p = dict(STREAK)
    if params:
        p.update(params)
    th = float(p["opaque"])
    if t.ndim == 2:
        opaque = t <= th
    else:
        # Opaque in EVERY channel, three compares rather than a luma matmul --
        # same answer for a body the tracer killed outright, a third of the
        # cost, and it cannot be fooled by a strongly coloured material that
        # happens to average dark.
        opaque = (t[..., 0] <= th) & (t[..., 1] <= th) & (t[..., 2] <= th)
    k = int(p["min_width"]) | 1                       # odd, so the box centres
    core = _wide_opaque(opaque, k)
    if int(core.sum()) < 4 * k * k:                   # no pin worth drawing on
        return None

    fit = _pin_axis(core, k)
    if fit is None:
        return None
    x0, y0, ax, ay, half_w, half_l, axis_certain, clip_lo, clip_hi = fit
    # The aspect test exists to reject a body whose ORIENTATION cannot be
    # trusted -- a compact blob, where the fit is a handful of near-square
    # slices and the answer is whichever way the noise fell.  1.8 because a pin
    # only ever enters from one side and what is in frame is a stub of shank:
    # 2.87 on the shipped hampton frame and 2.0 on E02, against 4.1 on A01.
    #
    # It is SKIPPED when the frame cleanly severed the body, because then the
    # orientation is known for certain from the sever and the visible aspect is
    # just an artefact of the crop.  Requiring 1.8 of a zoomed-in pin is what
    # made the glint disappear past ~1.5x: at 2.77x the in-frame piece is
    # 211 px wide and 59 px long, an aspect of 0.28, and still obviously a pin.
    if not axis_certain and half_l < float(p["min_aspect"]) * half_w:
        return None
    # A body WIDER THAN THE FRAME'S SHORT SIDE is not a pin whose sides we can
    # see -- it is something the crop is looking inside of, and half_w is then
    # a property of the crop rather than of the object.  This is what keeps
    # `mitegen_200um` off (its 1 um pixels fit a 528 px body in a 480 px frame,
    # and it would otherwise draw at half its angles and blink), and it is the
    # ONLY separator that survived: aspect does not work (the zoomed pin is
    # 0.28 against mitegen's 0.85) and neither does bar-likeness (35% against
    # 57% -- the zoomed pin is the LESS bar-like of the two).
    if 2.0 * half_w > float(p["max_width"]) * min(opaque.shape):
        return None
    ys, xs = np.nonzero(core)
    dy, dx = ys - y0, xs - x0

    # Work in the pin's bounding box, and inside it only in the ridge's own
    # band.  The exponential and the grain are the whole cost of this function
    # and both are pointless where the ridge has already fallen to nothing:
    # restricting them takes it from ~10.6 ms to ~2 ms on a 640x480 frame,
    # which matters because it runs on every served frame and the worst-case
    # /motor stream has no headroom to give away.
    #
    # `ys`/`xs` come from the ERODED mask, so the real body reaches (k-1)/2
    # further on every side; k of padding covers that and nothing more.  The
    # ridge lives inside the pin, so padding out by half_w (as an earlier
    # version did) tripled the box for no pixels.
    h, w = opaque.shape
    r0, r1 = max(int(ys.min()) - k, 0), min(int(ys.max()) + k + 1, h)
    c0, c1 = max(int(xs.min()) - k, 0), min(int(xs.max()) + k + 1, w)
    rr = np.arange(r0, r1, dtype=np.float64) - y0
    cc = np.arange(c0, c1, dtype=np.float64) - x0
    sigma = max(float(p["width"]) * 2.0 * half_w / 2.3548200450309493, 1e-6)

    # Signed distance across the pin from the ridge's centre-line.  Cut at
    # 3.5 sigma: the Gaussian is 2.2e-3 there, which after the gain and the
    # field is 0.1 grey levels -- under the quantiser, so the cut cannot draw
    # an edge of its own.
    d = (rr * ax)[:, None] - (cc * ay)[None, :] - float(p["offset"]) * half_w
    band = (np.abs(d) < 3.5 * sigma) & opaque[r0:r1, c0:c1]
    if not band.any():
        return None
    bi, bj = np.nonzero(band)
    dv = d[bi, bj]
    u = cc[bj] * ax + rr[bi] * ay                     # along the pin

    ridge = np.exp(-0.5 * (dv / sigma) ** 2)
    # Roll off an end that is really an end, so the streak does not stop in a
    # hard line at the bevelled tip -- but never an end the frame cut, where
    # the shank continues and the crop is the only reason it stopped.  A01
    # tapers toward the tip and E02 does not, so the roll-off is a rendering
    # choice, not a fit.
    tail = max(0.1 * half_l, 1e-6)
    e = np.ones_like(u)
    if not clip_hi:
        e = np.minimum(e, np.clip((half_l - u) / tail, 0.0, 1.0))
    if not clip_lo:
        e = np.minimum(e, np.clip((half_l + u) / tail, 0.0, 1.0))
    ridge *= e * e * (3.0 - 2.0 * e)
    # SCINTILLATION, not a texture glued to the pin.  The grain used to be
    # hashed on the pin's own frame so it would ride with it -- which is right
    # for a static surface pattern and WRONG for what this actually is.  A
    # machined shank is rough at the scale of the wavelength, so as it turns,
    # different micro-facets come into the specular condition and the glint
    # twinkles rather than translating rigidly.  Anchoring it also made the
    # pattern slide against the pin whenever the visible portion changed, which
    # read as parallax and gave the whole thing away as painted on.
    #
    # `phase` re-rolls the pattern; the server derives it from the POSE, so it
    # is still a pure function of what is being rendered -- a frame re-rendered
    # at the same pose is byte-identical, which `test_server_settle_parity`
    # requires, and a held pose does not shimmer.
    ridge *= 1.0 + float(p["grain"]) * _value_noise(
        u, dv, float(p["grain_px"]), int(p["seed"]) ^ int(p["phase"]))

    # `band` already carries the opaque mask, so the ridge is only ever drawn
    # where the tracer said the body is opaque: it cannot leak onto the
    # background or onto the loop.
    return bi + r0, bj + c0, np.maximum(ridge, 0.0) * float(p["gain"])


def specular_streak(t, params=None):
    """Additive specular term for the pin, in units of the local illumination.

    `t` is the transmittance image, (H, W) or (H, W, 3), on the delivered pixel
    grid.  Returns an (H, W) float array to be scaled by the illumination and
    added to the observed image -- all zero if the frame holds no opaque body
    at least `min_width` across, which is the case for a scene with no pin in
    view and is why this is safe to leave on.

    The geometry comes from the IMAGE, not from the scene: second moments of
    the eroded opaque mask give the pin's centroid, its long axis and its
    width, so the streak follows the pin through any spindle angle, pan or
    zoom without this stage knowing anything about the goniometer -- which is
    also what keeps it out of the tracer and off the templates.  ONE body is
    assumed: the erosion leaves only the pin in every scene that ships, but
    two separated wide bodies would be averaged into a single nonsense axis.
    """
    shape = t.shape if t.ndim == 2 else t.shape[:2]
    out = np.zeros(shape, dtype=np.float64)
    patch = _streak_patch(t, params)
    if patch is not None:
        out[patch[0], patch[1]] = patch[2]
    return out


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
                 mono=False, enabled=True, streak=True, streak_params=None):
    """Map transmittance to what the camera would record.  Returns a new array.

    `img` is (H, W, 3) float in [0, 1] straight out of the tracer, or a crop of
    a template holding the same quantity.  The result is in [floor, level*max
    (shape)] and therefore never touches 0 or 255 after quantisation -- the
    specular streak is bounded well below the white rail by its own gain, so
    adding it does not put the top rail back in reach.

    `streak` adds the pin's specular glint (`specular_streak`).  It reads the
    TRANSMITTANCE, before the mono collapse, because the pin is opaque in every
    channel and the geometry must not depend on whether colour was flattened.
    """
    if not enabled:
        return img
    t = to_luma(img) if mono else img
    e = float(level) * vignette(t.shape[0], t.shape[1], coeffs, amplitude)
    out = (e[..., None] - float(floor)) * t + float(floor)
    if streak:
        patch = _streak_patch(img, streak_params)
        if patch is not None:
            # Scaled by the local illumination: a specular return is reflected
            # incident light, so it dims where the field dims, exactly as the
            # background does.  Scattered into the pin's own pixels rather than
            # added frame-wide -- the streak covers ~6% of a frame.
            r, c, val = patch
            out[r, c, :] += (e[r, c] * val)[:, None]
    return out
