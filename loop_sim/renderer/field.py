"""Camera emulation: sensor raster, illumination field, black floor, tone.

The ray tracer outputs transmittance T (1.0 = empty field of view, 0.0 =
opaque); this module adds what a camera actually records:

    observed = (E(x,y) - B) * T(x,y) + B

`E` is the incident illumination field (what the camera sees at T=1); `B` is
the black floor (what it sees at T=0: veiling glare in the objective plus the
sensor pedestal).  See docs/DECISIONS.md 2026-08-10 renders became
photographs.

Runs at serve time only, downstream of both tracers, and must never move into
either or into a template.  The illumination defect is fixed in camera space,
so a template baking it in would pan with the sample instead of staying
still.  `frame_library.content_window`'s sample/background split would read
it as content everywhere and refuse to build.  And templates store raw
transmittance, so moving it here would force a rebuild of every shipped
library for a change that never touches them.

Everything here is a pure function of (shape, parameters): no RNG, no clock,
no per-call state.  `tests/test_server_settle_parity.py` and
`tests/test_torch_render_parity.py` compare JPEG bytes across a fresh render
vs. the live server, and across engines; a future term needing grain must
derive from pixel coordinates or a stored tile, never `np.random` without a
seed threaded through the call.

`E0`, the vignette shape and `B` were fitted to an empty field reconstructed
from 300 frames of one 2020-01-29 session, by the per-pixel 80th percentile;
the quadratic explains 83.5% of the field's variance.
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

# What the quadratic vignette above leaves behind: the soft blotchiness every
# real frame has.  Fitting only the smooth bowl and discarding this residual
# is why an earlier version of this stage read as a flat grey card next to a
# photograph.  Deterministic and fixed in camera space like the bowl, since
# this is an illumination defect too (the sample moves under it); that is the
# opposite of the pin's grain, which belongs to the pin and re-rolls with the
# pose -- the two look similar in code and are physically unrelated.
#
# Six octaves, not the usual one or two, because real background energy is
# roughly flat across scale rather than concentrated at one; gain 0.90 (not
# the textbook 0.5) for the same reason.  Total amplitude is 3.6% of level,
# from a background mask that dilates the dark body rather than thresholding
# by percentile.  See docs/DECISIONS.md 2026-08-11 the glint met an operator.
#
# The shipped default matches the 2020 session it was fit to, not a universal
# field (docs/DECISIONS.md 2026-08-10 renders became photographs).  A future
# caller wanting per-frame variation should vary these coefficients, not add
# noise -- a background fixed in camera space is exactly what a model could
# learn to localise by instead of the loop.
MOTTLE = 0.0435        # raw fBm amplitude, solved so 0.036 survives a quadratic re-fit
MOTTLE_UV = 0.80        # coarsest octave, in u-units (u spans 2.0 across w)
MOTTLE_OCTAVES = 6      # six, because one scale reads as a smooth wash, not cloud
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

# The real sensor's raster.  Every frame in `data/real_images/` is 704x480, and the
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
    flatter recent epochs (see docs/DECISIONS.md 2026-08-10 renders became
    photographs).

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
# A 0.7 mm mounting pin is machined steel; the tracer models it as a purely
# opaque body, so it renders as a flat silhouette at the black floor.  Real
# ones carry a bright, broken specular glint along the shank -- the largest
# remaining structural difference between a rendered pin and a photographed
# one.  Modelling it properly needs a BRDF and a specular bounce in the
# tracer; this is the cosmetic stand-in, deliberately in camera space so it
# costs no library rebuild.  The grain applies multiplicatively to the ridge,
# not to the diffuse floor: surface slope modulates what is reflected, not
# what is absorbed, and keeping it on the ridge alone stops it touching the
# pin body.
#
# Constants were measured off two reference frames at the mid zoom stop
# (`data/real_images/mid_mag/A01_nylonloop_pinleft_mid.jpg`,
# `E02_digitize_source_mid.jpg`); see docs/DECISIONS.md 2026-08-10 the pin's
# specular streak.  `gain` sits deliberately near the quiet end of the
# measured range: a blown-out glint on an otherwise flat pin reads worse than
# no glint at all.  Not modelled: a bright rim at the pin's far edge (edge
# diffraction plus the cylinder's grazing return) -- a second term, and much
# less visible than the streak.
#
# Geometry (`pin`, below) comes from the scene via
# `pin_projection.project_pin`, not from the image; see docs/DECISIONS.md
# 2026-08-12 the glint is projected from the scene.
STREAK = {
    "opaque":     0.04,   # transmittance at or below which a pixel is opaque
    "offset":    -0.45,   # ridge centre, in half-widths from the pin's axis;
                          # sign follows the illumination geometry, not the
                          # pin (default follows E02, this repo's digitized
                          # frame)
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


def _gaussian_kernel(sigma):
    """Normalised 1-D Gaussian, truncated at 3 sigma, cached by sigma.

    Separable, so a 2-D blur is two passes of this.  Deterministic and
    dependency-free: `optics.py` owns the objective PSF and PIL owns the
    template's defocus, but neither can be reached from here without either a
    circular import or a PIL round-trip through uint8.
    """
    key = round(float(sigma), 4)
    hit = _weight_cache.get(("gauss", key))
    if hit is not None:
        return hit
    r = max(int(3.0 * key + 0.5), 1)
    x = np.arange(-r, r + 1, dtype=np.float64)
    k = np.exp(-0.5 * (x / max(key, 1e-9)) ** 2)
    k /= k.sum()
    _weight_cache[("gauss", key)] = k
    return k


def _blur1d(a, k, axis):
    """One separable pass, as a weighted sum of SHIFTED copies.

    Not `np.apply_along_axis(np.convolve, ...)`: that runs a Python-level call
    per row, and on a defocused frame it cost 9.5 ms where this costs under 1.
    Here the whole array moves at once, once per kernel tap.
    """
    r = (len(k) - 1) // 2
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    p = np.pad(a, pad, mode="edge")
    n = a.shape[axis]
    out = np.zeros_like(a)
    sl = [slice(None)] * a.ndim
    for i, kv in enumerate(k):
        sl[axis] = slice(i, i + n)
        out += kv * p[tuple(sl)]
    return out


def _blur2d(a, sigma):
    """Separable Gaussian blur of a 2-D array, edge-extended."""
    k = _gaussian_kernel(sigma)
    return _blur1d(_blur1d(a, k, 1), k, 0)


def _streak_patch(t, pin, params=None, defocus=0.0):
    """`(rows, cols, values)` for the pin's glint, or None if there is no pin.

    The sparse form.  `specular_streak` is the dense wrapper; `apply_camera`
    uses this one so a frame whose streak covers ~20k pixels does not pay a
    337k-pixel multiply and add to deliver it.  One implementation, two
    shapes -- the discipline the JPEG-byte guards need.

    `pin` is `(x0, y0, ax, ay, half_w, half_l, clip_lo, clip_hi)` in the pixels
    of `t`, from `pin_projection.project_pin`: the centre of the visible shank,
    its unit direction, its half-width and half-length, and which ENDS the frame
    cut.  **`pin is None` means there is no pin on this frame and nothing is
    drawn** -- that is the answer for a pose that has panned or zoomed past it,
    for a mount seen end-on, and for a scene with no shiny body.

    Geometry comes from the scene, not the image: fitting a bar to whatever
    dark pixels survived erosion painted the glint onto the droplet once the
    pin left frame, and no image-only rule separates the two bodies.  See
    docs/DECISIONS.md 2026-08-12 the glint is projected from the scene.
    """
    if pin is None:
        return None
    p = dict(STREAK)
    if params:
        p.update(params)
    x0, y0, ax, ay, half_w, half_l, clip_lo, clip_hi = pin
    if half_w <= 0.0 or half_l <= 0.0:
        return None

    # Work in the shank's bounding box, and inside it only in the ridge's own
    # band.  The exponential and the grain are the whole cost of this function
    # and both are pointless where the ridge has already fallen to nothing:
    # restricting them takes it from ~10.6 ms to ~2 ms on a 640x480 frame,
    # which matters because it runs on every served frame and the worst-case
    # /motor stream has no headroom to give away.
    #
    # The box is the projected rectangle -- the axis segment swept by half_w to
    # either side -- plus a pixel of slack for the half-open comparisons.  It no
    # longer comes from a mask, so nothing has to be padded back out by an
    # erosion radius.
    h, w = t.shape[:2]
    ex = abs(ax) * half_l + abs(ay) * half_w + 1.0
    ey = abs(ay) * half_l + abs(ax) * half_w + 1.0
    r0, r1 = max(int(np.floor(y0 - ey)), 0), min(int(np.ceil(y0 + ey)) + 1, h)
    c0, c1 = max(int(np.floor(x0 - ex)), 0), min(int(np.ceil(x0 + ex)) + 1, w)
    if r1 <= r0 or c1 <= c0:
        return None

    tile = t[r0:r1, c0:c1]
    th = float(p["opaque"])
    if tile.ndim == 2:
        opaque = tile <= th
    else:
        # Opaque in EVERY channel, three compares rather than a luma matmul --
        # same answer for a body the tracer killed outright, a third of the
        # cost, and it cannot be fooled by a strongly coloured material that
        # happens to average dark.
        opaque = ((tile[..., 0] <= th) & (tile[..., 1] <= th)
                  & (tile[..., 2] <= th))

    rr = np.arange(r0, r1, dtype=np.float64) - y0
    cc = np.arange(c0, c1, dtype=np.float64) - x0
    sigma = max(float(p["width"]) * 2.0 * half_w / 2.3548200450309493, 1e-6)

    # Signed distance ACROSS the shank from its axis, and distance ALONG it.
    dperp = (rr * ax)[:, None] - (cc * ay)[None, :]
    u2 = (cc * ax)[None, :] + (rr * ay)[:, None]
    # ... and from the ridge's own centre-line, which sits `offset` half-widths
    # off the axis.  Cut at 3.5 sigma: the Gaussian is 2.2e-3 there, which after
    # the gain and the field is 0.1 grey levels -- under the quantiser, so the
    # cut cannot draw an edge of its own.
    d = dperp - float(p["offset"]) * half_w
    # THREE conditions, and the last two are what stop the leak.  The ridge is
    # drawn only where the tracer says the body is opaque AND inside the shank
    # the scene projected.  Bounding by `opaque` alone was the second half of
    # the droplet bug: the band ran the width of the frame and landed on every
    # dark pixel it crossed, which is how the loop fiber and the drop's rim
    # picked up a glint the fit had specifically excluded.
    band = ((np.abs(d) < 3.5 * sigma) & (np.abs(dperp) <= half_w)
            & (np.abs(u2) <= half_l) & opaque)
    if not band.any():
        return None
    bi, bj = np.nonzero(band)
    dv = d[bi, bj]
    u = u2[bi, bj]

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
    # A machined shank is rough at the scale of the wavelength: as it turns,
    # different micro-facets enter the specular condition, so the glint
    # scintillates rather than translating like a texture glued to the pin.
    #
    # `phase` re-rolls the pattern; the server derives it from the pose, so it
    # is still a pure function of what is being rendered -- a frame re-rendered
    # at the same pose is byte-identical, which `test_server_settle_parity`
    # requires, and a held pose does not shimmer.
    ridge *= 1.0 + float(p["grain"]) * _value_noise(
        u, dv, float(p["grain_px"]), int(p["seed"]) ^ int(p["phase"]))

    ridge = np.maximum(ridge, 0.0) * float(p["gain"])

    # DEFOCUS.  The glint is light from the pin's SURFACE, so when the sample
    # sits off the focal plane it blurs with everything else on that plane --
    # it does not stay razor-sharp on a pin that has visibly gone soft.  Same
    # sigma the template crop applies to the silhouette (`pose_crop`), so the
    # two cannot disagree.
    #
    # Applied AFTER the shank mask on purpose: a defocused glint spreads a
    # little past the pin's edge, exactly as the blurred silhouette does.  And
    # applied to the GLINT ALONE, not to the whole frame -- swapping the stage
    # order instead would also soften the illumination field, whose finest
    # octave is ~9 px against a sigma that reaches 4.7 px at 1 mm of depth, and
    # the background is not imaged from the sample plane so it must not move.
    if defocus and defocus > 0.05:
        pad = int(3.0 * defocus + 1.5)
        dense = np.zeros((r1 - r0 + 2 * pad, c1 - c0 + 2 * pad))
        dense[bi + pad, bj + pad] = ridge
        dense = _blur2d(dense, float(defocus))
        R0, C0 = max(r0 - pad, 0), max(c0 - pad, 0)
        dense = dense[R0 - (r0 - pad):dense.shape[0] - max(r1 + pad - h, 0),
                      C0 - (c0 - pad):dense.shape[1] - max(c1 + pad - w, 0)]
        gi, gj = np.nonzero(dense > 1e-6)
        if gi.size == 0:
            return None
        return gi + R0, gj + C0, dense[gi, gj]
    return bi + r0, bj + c0, ridge


def specular_streak(t, pin, params=None, defocus=0.0):
    """Additive specular term for the pin, in units of the local illumination.

    `t` is the transmittance image, (H, W) or (H, W, 3), on the delivered pixel
    grid.  Returns an (H, W) float array to be scaled by the illumination and
    added to the observed image -- all zero when `pin` is None, which is the
    case for a scene with no pin in view and is why this is safe to leave on.

    The geometry comes from the SCENE, not from the image: `pin` is the shank's
    axis, half-width and half-length in these pixels, projected by
    `pin_projection.project_pin` from the object the scene declares shiny and
    the pose the server is serving.  That makes it exact at any zoom, crop or
    spindle angle, and it is the whole reason the glint can no longer land on
    the droplet.  This stage still never sees the goniometer itself, so it
    stays numpy-only, out of both tracers and off the templates.
    """
    shape = t.shape if t.ndim == 2 else t.shape[:2]
    out = np.zeros(shape, dtype=np.float64)
    patch = _streak_patch(t, pin, params, defocus)
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
    library rebuild.

    Off by default: the simulator is a colour instrument, and a delivery-stage
    flatten hid the scene bug instead of exposing it.  On the shipped libraries
    the difference is at most 20/21/46 levels on 0.003-0.42% of pixels
    (realistic/hampton/mitegen), a tint on loop and droplet edges.  The option
    stays for comparisons against a monochrome camera.
    """
    return np.repeat((img @ _LUMA)[..., None], 3, axis=2)


def apply_camera(img, level=E0, floor=B, coeffs=VIGNETTE, amplitude=1.0,
                 mono=False, enabled=True, streak=True, streak_params=None,
                 defocus=0.0, pin=None):
    """Map transmittance to what the camera would record.  Returns a new array.

    `img` is (H, W, 3) float in [0, 1] straight out of the tracer, or a crop of
    a template holding the same quantity.  The result is in [floor, level*max
    (shape)] and therefore never touches 0 or 255 after quantisation -- the
    specular streak is bounded well below the white rail by its own gain, so
    adding it does not put the top rail back in reach.

    `streak` adds the pin's specular glint (`specular_streak`), and `pin` says
    WHERE the pin is -- see `_streak_patch`.  `pin=None` draws no glint, which
    is the right default for any caller that does not know where the pin is: a
    frame with no glint is a small incorrectness, a glint on the wrong body is
    a false feature in training data.  The glint reads the TRANSMITTANCE,
    before the mono collapse, because the pin is opaque in every channel and
    the result must not depend on whether colour was flattened.  `defocus` is
    the sigma the sample's silhouette was blurred by, in output pixels; the
    glint gets the same blur, because it comes off the same surface.  The
    illumination field deliberately does NOT.
    """
    if not enabled:
        return img
    t = to_luma(img) if mono else img
    e = float(level) * vignette(t.shape[0], t.shape[1], coeffs, amplitude)
    out = (e[..., None] - float(floor)) * t + float(floor)
    if streak:
        patch = _streak_patch(img, pin, streak_params, defocus)
        if patch is not None:
            # Scaled by the local illumination: a specular return is reflected
            # incident light, so it dims where the field dims, exactly as the
            # background does.  Scattered into the pin's own pixels rather than
            # added frame-wide -- the streak covers ~6% of a frame.
            r, c, val = patch
            out[r, c, :] += (e[r, c] * val)[:, None]
    return out
