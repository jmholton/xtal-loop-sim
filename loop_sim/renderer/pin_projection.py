"""Where the pin is on the delivered frame, projected from the scene.

`field.py` draws a cosmetic specular glint along the mounting pin; this
module supplies its geometry so the glint tracks the scene instead of being
inferred from the rendered picture (see docs/DECISIONS.md 2026-08-12 the
glint is projected from the scene).

`pin_segment(scene)` returns the visible shank's `(A, B, radius)` in scene
mm.  `project_pin(scene, gonio, to_px, frame_wh)` returns
`(x0, y0, ax, ay, half_w, half_l, clip_lo, clip_hi)` in delivered pixels --
the shank's centre, unit axis, half-width, half-length, and which ends the
frame cut -- or None if there is no pin to draw (past the frame, end-on, or
wider than `MAX_WIDTH` of the frame).  The camera is orthographic, so a
cylinder's silhouette half-width is its radius at any tilt short of end-on.

Must live in `renderer/`, not `scene/*.py` or `motors/goniometer.py`, and must
never read a flag from scene YAML: both are hashed into `render_sha` /
`scene_sha256` as build keys, and a change here must not invalidate a shipped
frame library.  `SHINY` below is the code-level declaration this forces.
"""
import numpy as np

from ..motors.goniometer import apply_transform

# Bodies the glint models, as (object name, material name).  The stand-in is
# calibrated on `data/real_images/mid_mag/A01` and `E02`, both of which show a machined steel
# shank; nothing else in these scenes returns a specular highlight.  A kapton
# micromount is DECLARED not to shine rather than guessed at from its shape,
# which is what the old `max_width` heuristic was doing.
SHINY = {("pin", "metal")}

# Below this, the shank points too near the view axis to show a side at all --
# it projects as a disc, and there is no length to run a ridge along.  The value
# is sin(angle from the optical axis): 0.15 is ~8.6 degrees, by which point the
# shank is foreshortened more than 6:1.
EDGE_ON = 0.15

# A shank wider than this fraction of the frame's short side gets no glint.
# This is a stated DOMAIN LIMIT of the cosmetic model, not a shape guess: the
# ridge's position and FWHM are both defined as fractions of the pin's width
# (`STREAK["offset"]`, `STREAK["width"]`), so they are undefined when the width
# is not in frame.  Carried over from the heuristic it replaces, where it was
# the only separator that survived; the input is now exact instead of measured
# off a silhouette.  It is what keeps `mitegen_200um` off: its pin is
# `axis [0,0,1]` at 1 um pixels, so it projects 500 px wide against a 480-row
# frame at every angle, and end-on at phi=0.
MAX_WIDTH = 0.90


def _unwrap_cylinder(shape):
    """`(cylinder, [half_spaces])` from a pin shape, or `(None, [])`.

    The pin is never a bare `Cylinder` in any shipped scene -- it is
    `Intersection(Cylinder, HalfSpace)`, where the half-space is the bevel that
    `crystal_harvester.pin_geometry.make_pin` cuts across the tip.  The
    half-spaces are what tell us where the shank actually STARTS, which is the
    job the old `axis_gate` heuristic was doing by dropping narrow slices.
    """
    cyl, cuts = None, []
    stack = [shape]
    while stack:
        s = stack.pop()
        if hasattr(s, "children"):                  # Intersection / Union / ...
            stack.extend(s.children)
        elif hasattr(s, "radius") and hasattr(s, "height") and hasattr(s, "axis"):
            if cyl is None:
                cyl = s
        elif hasattr(s, "normal") and hasattr(s, "offset"):
            cuts.append(s)
    return cyl, cuts


def pin_segment(scene):
    """`(A, B, radius)` in SCENE mm for the shiny shank, or None.

    `A` and `B` are the two ends of the visible shank ON ITS AXIS: the
    cylinder's own extent, clipped by every half-space that cuts it.  For
    `hampton_300um_realistic` that turns the cylinder's x = 0.7 .. 6.7 into
    x = 1.0 .. 6.7, which is where the metal actually begins.
    """
    for obj in scene.objects:
        mat = getattr(obj.material, "name", obj.material)
        if (obj.name, mat) not in SHINY:
            continue
        cyl, cuts = _unwrap_cylinder(obj.shape)
        if cyl is None:
            return None
        half = 0.5 * cyl.height
        lo, hi = -half, half                        # arclength from cyl.centre
        for hs in cuts:
            # Interior is dot(n, p) <= offset, and p(s) = centre + s * axis, so
            # the cut is a bound on s -- an upper one if the axis runs with the
            # normal, a lower one if against it.
            d = float(hs.normal @ cyl.axis)
            r = float(hs.offset - hs.normal @ cyl.centre)
            if abs(d) < 1e-12:                      # cut is parallel to the axis
                if r < 0.0:
                    return None                     # ... and removes all of it
                continue
            b = r / d
            if d > 0.0:
                hi = min(hi, b)
            else:
                lo = max(lo, b)
        if hi - lo <= 1e-9:
            return None
        return (cyl.centre + cyl.axis * lo,
                cyl.centre + cyl.axis * hi,
                float(cyl.radius))
    return None


def _clip_to_frame(p, q, w, h, mx, my):
    """Clip segment p->q to the frame inflated by `(mx, my)`, Liang-Barsky.

    Returns `(p', q', cut_lo, cut_hi)` or None if the segment misses entirely.
    The flags say which END the FRAME cut, which is what decides whether the
    ridge tapers there: a bevelled tip is an end and rolls off, a crop is not
    and must run straight off the edge.

    The margin is the ridge's reach PERPENDICULAR to the axis, resolved onto
    each pixel axis by the caller -- a pin whose axis is just past the top edge
    can still have its lower flank, and its ridge, inside the frame.  It must
    NOT be applied along the axis: the shank stops where it stops, and
    inflating that direction keeps a pin "in frame" for a third of a millimetre
    after it has left (measured: it held `hampton_300um_realistic` alive at zoom
    2.5 and 3.0, where the crop ends 0.2 mm short of the metal).
    """
    x0, y0 = p
    dx, dy = q[0] - x0, q[1] - y0
    lo, hi = 0.0, 1.0
    for num, den in ((x0 + mx, -dx), (w - 1.0 + mx - x0, dx),
                     (y0 + my, -dy), (h - 1.0 + my - y0, dy)):
        if abs(den) < 1e-12:
            if num < 0.0:
                return None                         # parallel and outside
            continue
        t = num / den
        if den > 0.0:
            hi = min(hi, t)
        else:
            lo = max(lo, t)
        if lo > hi:
            return None
    return ((x0 + lo * dx, y0 + lo * dy),
            (x0 + hi * dx, y0 + hi * dy),
            lo > 0.0, hi < 1.0)


def project_pin(scene, gonio, to_px, frame_wh):
    """`(x0, y0, ax, ay, half_w, half_l, clip_lo, clip_hi)` in delivered px.

    None means "no pin to draw on this frame", which is the answer for a pose
    that has panned or zoomed past it, for a scene with no shiny body, and for
    a mount seen end-on.  `field._streak_patch` draws nothing when it gets None.

    `to_px` maps a point on the image plane, in mm along `camera_fast` and
    `camera_slow`, to a column and row on the DELIVERED grid -- see
    `camera_mapper` and `template_mapper`.  Keeping it a callable is what lets
    the live path and the template path share every line of the geometry.

    Verified against the silhouette fit it replaces; see docs/DECISIONS.md
    2026-08-12 the glint is projected from the scene.
    """
    seg = pin_segment(scene)
    if seg is None:
        return None
    A, B, radius = seg

    g = scene.geometry
    fast = np.asarray(g.get("camera_fast", [1, 0, 0]), dtype=float)
    slow = np.asarray(g.get("camera_slow", [0, 1, 0]), dtype=float)

    lab = apply_transform(gonio.transform(), np.array([A, B], dtype=float))
    ua, va = float(lab[0] @ fast), float(lab[0] @ slow)
    ub, vb = float(lab[1] @ fast), float(lab[1] @ slow)

    # Length ON THE IMAGE PLANE, in mm.  The shank's true length times the sine
    # of its angle to the view axis, so this IS the foreshortening test -- no
    # separate dot with the optical axis is needed, and it stays correct for a
    # geometry whose axes are not the default orthonormal set.
    du, dv = ub - ua, vb - va
    span = (du * du + dv * dv) ** 0.5
    full = float(np.linalg.norm(B - A))
    if full <= 0.0 or span <= EDGE_ON * full:
        return None

    # The silhouette edge is the axis offset by the radius, perpendicular ON THE
    # IMAGE PLANE -- true for a cylinder at any tilt short of end-on.  Both the
    # axis and the edge go through `to_px`, so the half-width comes out in
    # delivered pixels with the sensor's 1.110 non-square aspect already in it
    # rather than applied afterwards as a scalar.
    nu, nv = du / span, dv / span
    ca, ra = to_px(ua, va)
    cb, rb = to_px(ub, vb)
    ce, re = to_px(ua - nv * radius, va + nu * radius)

    aln = ((cb - ca) ** 2 + (rb - ra) ** 2) ** 0.5
    if aln <= 1e-9:
        return None
    ax, ay = (cb - ca) / aln, (rb - ra) / aln
    half_w = abs(ax * (re - ra) - ay * (ce - ca))
    if half_w <= 0.0:
        return None

    w, h = frame_wh
    if 2.0 * half_w > MAX_WIDTH * min(w, h):
        return None

    # The ridge reaches at most `half_w` off the axis, and only PERPENDICULAR
    # to it, so resolve that reach onto the two pixel axes before inflating.
    clipped = _clip_to_frame((ca, ra), (cb, rb), w, h,
                             abs(ay) * half_w, abs(ax) * half_w)
    if clipped is None:
        return None
    (c0, r0), (c1, r1), cut_lo, cut_hi = clipped
    half_l = 0.5 * (((c1 - c0) ** 2 + (r1 - r0) ** 2) ** 0.5)
    if half_l <= 0.0:
        return None
    return (0.5 * (c0 + c1), 0.5 * (r0 + r1), ax, ay,
            half_w, half_l, bool(cut_lo), bool(cut_hi))


def camera_mapper(camera_cfg, zoom=1.0, sensor=None):
    """Image plane (mm) -> delivered pixels, for a LIVE render.

    `microscope.py` samples camera pixel i at `(i - W/2) * eff_px`, so the
    inverse is `u / eff_px + W/2`.  `sensor` is `field.SENSOR_WH` when the frame
    is being resampled onto the camera's 704x480 raster -- that runs BEFORE
    `apply_camera`, so the glint is drawn on the wide grid and the columns carry
    the 704/640 scale.  Same tax `_command_recenter` pays in the other
    direction.
    """
    W = int(camera_cfg.get("width", 640))
    H = int(camera_cfg.get("height", 480))
    eff = float(camera_cfg.get("pixel_size", 0.005)) / float(zoom)
    sx = (sensor[0] / float(W)) if sensor else 1.0
    sy = (sensor[1] / float(H)) if sensor else 1.0

    def to_px(u, v):
        return ((u / eff + 0.5 * W) * sx, (v / eff + 0.5 * H) * sy)

    return to_px, (sensor if sensor else (W, H))


def template_mapper(manifest, box, out_size, sensor=None):
    """Image plane (mm) -> delivered pixels, for a TEMPLATE replay.

    Three stages, composed: the rendered window maps mm to template pixels, the
    `box` and `out_size` `pose_crop` returned map template pixels to output
    pixels (that is where pan and zoom live), and `to_sensor` stretches the
    columns.

    NOTE the goniometer handed to `project_pin` alongside this mapper must carry
    the SPINDLE ROTATION ONLY.  The template already has phi baked into its
    pixels, and `box` already accounts for the translation and the zoom -- pass
    the pose's translations here as well and they are applied twice.
    """
    win = manifest["window_mm"]
    tpl_px = float(manifest["rendered"]["pixel_size"])
    W, H = int(out_size[0]), int(out_size[1])
    sx = (sensor[0] / float(W)) if sensor else 1.0
    sy = (sensor[1] / float(H)) if sensor else 1.0
    kx = W / float(box[2] - box[0])
    ky = H / float(box[3] - box[1])

    def to_px(u, v):
        col = ((u - float(win["x0"])) / tpl_px - float(box[0])) * kx
        row = ((v - float(win["y0"])) / tpl_px - float(box[1])) * ky
        return (col * sx, row * sy)

    return to_px, (sensor if sensor else (W, H))
