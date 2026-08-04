"""
Pre-computed frame library: a rotation sweep rendered once and replayed.

Rationale
---------
The camera is orthographic, so the only motor that genuinely changes image
content is the spindle.  Everything else is an image-space transform of a
sufficiently large, sufficiently finely sampled master frame:

    frames rendered  = 360 / step_deg          (e.g. 360 at 1 deg)
    tx / ty / tz     = crop offset (see `pose_crop` -- the stage rides on the
                       spindle, so which motor is lateral depends on phi)
    zoom             = rescale, exact down from the supersampled master
    depth            = defocus, approximated by a Gaussian blur

Supersampling
-------------
`supersample` divides the rendered pixel size, so the master is sampled finer
than the camera.  Zooming out from it is exact decimation; zooming in past it
would be upsampling, so `supersample` is the hard ceiling on zoom.  4x is the
default because it is where sampling critically matches the NA 0.10 objective
(Rayleigh 3.35 um -> Nyquist 1.68 um/px against a 7.4 um native pixel).

Framing
-------
The render window is measured from the scene rather than being a uniform blow-up
of the camera.  A mount is long and thin -- the hampton pin runs to x=6.7 mm
against a 4.7 mm field -- so a symmetric margin centred on the goniometer origin
leaves most of the pin unrendered and panning scrolls in blank background.
`content_window` renders a coarse wide-field scout sweep, measures where the
image actually differs from background, and the sweep is then rendered with a
fixed `tx` offset that centres that window.  `tx` is parallel to the spindle
axis, so a constant tx offset is rotation-invariant and is exactly equivalent
to moving the camera.

Layout
------
    frame_library/<scene_stem>/manifest.json
    frame_library/<scene_stem>/rot_0000.jpg ...

The manifest records a SHA-256 of the scene YAML and the build parameters;
`ensure_library()` rebuilds when either changes.
"""
import hashlib
import json
import math
import os
import time

import numpy as np

# Anchored to the repo root rather than the caller's cwd: a library written to
# the wrong directory is a silently missing deliverable, since it has to land in git.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ROOT = os.path.join(_REPO_ROOT, "frame_library")
DEFAULT_STEP_DEG = 1.0
DEFAULT_SUPERSAMPLE = 4
DEFAULT_PAN_MM = 0.6
DEFAULT_N_COND = 7
DEFAULT_QUALITY = 90
DEFAULT_VRAM_FRACTION = 0.80
# Sanity guard on the measured render window, not a hardware limit: 4x
# supersample over a 10 x 5 mm window is ~14 Mpx, so anything past this means
# the content measurement went wrong.
MAX_TEMPLATE_MPX = 200.0

# Build parameters that change the pixels. A library whose manifest disagrees
# with the requested value of any of these is stale, not merely different.
_BUILD_KEYS = ("axis", "step_deg", "n_cond", "supersample", "pan_mm",
               "jpeg_quality")


def scene_fingerprint(scene_path):
    with open(scene_path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def library_dir(scene_path, root=DEFAULT_ROOT):
    stem = os.path.splitext(os.path.basename(scene_path))[0]
    return os.path.join(root, stem)


def load_manifest(lib_dir):
    path = os.path.join(lib_dir, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _write_manifest(lib_dir, manifest):
    """Write manifest.json atomically and force it to disk.

    The manifest is the last thing a build produces and the one file that makes
    the other 360 usable, so it is exactly the wrong thing to leave sitting in
    the page cache. A host crash after a long build otherwise loses it and
    orphans the whole library -- which happened once, costing a rebuild that
    was only avoided because the window is deterministic and could be
    recomputed. Temp-file + fsync + rename also means a crash mid-write can
    never leave a truncated manifest behind.
    """
    path = os.path.join(lib_dir, "manifest.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(manifest, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def build_params(axis="rotx", step_deg=DEFAULT_STEP_DEG, n_cond=DEFAULT_N_COND,
                 supersample=DEFAULT_SUPERSAMPLE, pan_mm=DEFAULT_PAN_MM,
                 quality=DEFAULT_QUALITY, **_ignored):
    """The pixel-affecting settings, keyed as the manifest stores them.

    One place to resolve defaults, so `is_current` cannot drift from
    `build_library` and start reporting every default build as stale (or, worse,
    every changed build as current).
    """
    return {"axis": axis, "step_deg": step_deg, "n_cond": n_cond,
            "supersample": supersample, "pan_mm": pan_mm,
            "jpeg_quality": quality}


def is_current(scene_path, lib_dir, **params):
    """True when a complete library matching the scene AND the requested build
    parameters is on disk.

    Comparing the scene hash alone is not enough: asking for a different
    supersample or step and silently getting the old library back would be
    indistinguishable from a correct build.
    """
    man = load_manifest(lib_dir)
    if man is None:
        return False
    if man.get("scene_sha256") != scene_fingerprint(scene_path):
        return False
    for key in _BUILD_KEYS:
        want = params.get(key)
        if want is not None and man.get(key) != want:
            return False
    if not all(os.path.exists(os.path.join(lib_dir, f["file"]))
               for f in man["frames"]):
        return False
    # Cheap guard against a half-overwritten library: check one frame really is
    # the size the manifest claims. pose_crop computes boxes from the manifest,
    # and PIL pads an over-large box with black rather than raising, so a size
    # mismatch would serve silently corrupt frames.
    try:
        from PIL import Image
        rnd = man["rendered"]
        with Image.open(os.path.join(lib_dir, man["frames"][0]["file"])) as im:
            if im.size != (int(rnd["width"]), int(rnd["height"])):
                return False
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------
def content_window(scene, tscene, axis="rotx", n_probe=8, fov_mult=4.0,
                   scout_div=2, eps=2.0 / 255.0, max_fov_mult=64.0,
                   progress=None):
    """Image-plane bounding box of visible content, in mm, over a full sweep.

    Renders a coarse wide-field scout at `n_probe` spindle angles and takes the
    union of the pixels that differ from background. Measuring the rendered
    image rather than the geometry means this works for CSG and half-spaces,
    which have no finite bounding box of their own.

    If content reaches the edge of the scout frame the field is widened and the
    measurement retried, because a truncated window is the silent version of
    exactly the bug this exists to prevent: panning would run off the rendered
    area into blank background.

    Returns (x0, x1, y0, y1) in mm about the goniometer origin.
    """
    from ..motors.goniometer import Goniometer
    from ..renderer.engine_torch import render_torch

    cam = scene.camera_cfg
    W0, H0 = int(cam["width"]), int(cam["height"])
    px0 = float(cam["pixel_size"])
    saved = (cam["width"], cam["height"], cam["pixel_size"])

    try:
        # Scout resolution is fixed and the pixel is coarsened to widen the
        # field, so doubling the survey costs nothing extra to render.
        sw, sh = max(32, W0 // scout_div), max(32, H0 // scout_div)
        while True:
            spx = px0 * fov_mult * scout_div
            cam["width"], cam["height"], cam["pixel_size"] = sw, sh, spx

            x0 = y0 = math.inf
            x1 = y1 = -math.inf
            clipped = False
            for i in range(n_probe):
                ang = 360.0 * i / n_probe
                gono = Goniometer(scene.geometry).set(**{axis: ang})
                img = render_torch(tscene, gono, n_cond=1).cpu().numpy()
                # Per channel: pooling them treats any tinted or per-channel
                # background as content everywhere, which reads as "content
                # fills the frame" and drives the widening loop to its cap.
                border = np.concatenate([img[0], img[-1]])
                bg = np.median(border, axis=0)                    # (3,)
                mask = (np.abs(img - bg) > eps).any(axis=2)
                if not mask.any():
                    continue
                rows, cols = np.where(mask)
                if (cols.min() == 0 or cols.max() == sw - 1
                        or rows.min() == 0 or rows.max() == sh - 1):
                    clipped = True
                x0 = min(x0, (cols.min() - sw / 2.0) * spx)
                x1 = max(x1, (cols.max() + 1 - sw / 2.0) * spx)
                y0 = min(y0, (rows.min() - sh / 2.0) * spx)
                y1 = max(y1, (rows.max() + 1 - sh / 2.0) * spx)

            if not clipped or fov_mult >= max_fov_mult:
                if clipped and progress:
                    progress(f"[frame-library] WARNING content still reaches the "
                             f"scout edge at {fov_mult:g}x the field of view -- the "
                             f"render window may be truncated")
                break
            fov_mult *= 2.0
            if progress:
                progress(f"[frame-library] content reaches the scout edge; "
                         f"widening the survey to {fov_mult:g}x the field of view")
    finally:
        cam["width"], cam["height"], cam["pixel_size"] = saved

    if not math.isfinite(x0):          # nothing visible; fall back to the camera
        return (-W0 * px0 / 2, W0 * px0 / 2, -H0 * px0 / 2, H0 * px0 / 2)
    return (x0, x1, y0, y1)


def _round_up_to_parity(n, like):
    """Smallest integer >= n with the same parity as `like`."""
    n = int(n)
    return n if (n - int(like)) % 2 == 0 else n + 1


def plan_window(cam, content, pan_mm=DEFAULT_PAN_MM):
    """Render window (x0, x1, y0, y1) in mm covering content, the centred
    field of view, and `pan_mm` of travel beyond both."""
    W, H = int(cam["width"]), int(cam["height"])
    px = float(cam["pixel_size"])
    fw, fh = W * px / 2.0, H * px / 2.0
    cx0, cx1, cy0, cy1 = content
    return (min(cx0, -fw) - pan_mm, max(cx1, fw) + pan_mm,
            min(cy0, -fh) - pan_mm, max(cy1, fh) + pan_mm)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_library(scene_path, root=DEFAULT_ROOT, axis="rotx",
                  step_deg=DEFAULT_STEP_DEG, supersample=DEFAULT_SUPERSAMPLE,
                  pan_mm=DEFAULT_PAN_MM, n_cond=DEFAULT_N_COND,
                  quality=DEFAULT_QUALITY, tile_size=None,
                  vram_fraction=DEFAULT_VRAM_FRACTION,
                  device=None, progress=print):
    """Render a full 360 deg sweep about `axis` and write it to disk.

    Raises RuntimeError on out-of-memory rather than silently degrading -- a
    library rendered at a reduced setting would be indistinguishable from a
    good one afterwards.
    """
    import torch
    from PIL import Image
    from ..scene.scene import load
    from ..motors.goniometer import Goniometer
    from ..renderer.engine_torch import TorchScene, render_torch

    if axis != "rotx":
        # build_library's window offset and pose_crop both hard-code the rotx
        # coupling (tx parallel to the spindle, ty/tz counter-rotated). Another
        # axis needs both generalised; failing loudly beats a wrong library.
        raise ValueError(
            f"axis={axis!r} is not supported yet -- the window offset and "
            f"pose_crop both assume the spindle is rotx. Generalise both "
            f"before enabling another axis.")

    lib_dir = library_dir(scene_path, root)
    os.makedirs(lib_dir, exist_ok=True)
    # Retire the old manifest FIRST. Frames are overwritten in place, so an
    # interrupted build would otherwise leave a manifest describing a mix of
    # old and new frames -- and is_current would call it good.
    man_path = os.path.join(lib_dir, "manifest.json")
    if os.path.exists(man_path):
        os.remove(man_path)

    scene = load(scene_path, device="cpu")
    cam = scene.camera_cfg
    W, H = int(cam["width"]), int(cam["height"])
    px0 = float(cam["pixel_size"])
    na_obj, na_cond = float(cam["na_objective"]), float(cam["na_condenser"])

    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    tscene = TorchScene(scene, dev, torch.float64)
    if progress:
        progress(f"[frame-library] device {dev}")

    # --- framing: measure the scene, then centre the window on it -----------
    content = content_window(scene, tscene, axis=axis, progress=progress)
    wx0, wx1, wy0, wy1 = plan_window(cam, content, pan_mm=pan_mm)
    tpl_px = px0 / supersample
    # Keep the template the same parity as the camera so (RW - W)/2 is a whole
    # number: an odd template puts every crop boundary on a half pixel and
    # costs a systematic 1-px wobble against a live render.
    RW = _round_up_to_parity(math.ceil((wx1 - wx0) / tpl_px), W)
    RH = _round_up_to_parity(math.ceil((wy1 - wy0) / tpl_px), H)
    # Refuse an absurd window rather than committing hours to it. A bad content
    # measurement (a scene whose background the scout cannot identify) shows up
    # here as a template orders of magnitude larger than the camera.
    if RW * RH > MAX_TEMPLATE_MPX * 1e6:
        raise RuntimeError(
            f"refusing to build a {RW}x{RH} ({RW*RH/1e6:.0f} Mpx) template for "
            f"{scene_path} -- over the {MAX_TEMPLATE_MPX:g} Mpx guard. The "
            f"measured content window was {wx1-wx0:.1f} x {wy1-wy0:.1f} mm "
            f"against a {W*px0:.1f} x {H*px0:.1f} mm field; if that looks wrong "
            f"the scout could not separate the sample from the background. "
            f"Lower --supersample or --pan-mm, or raise MAX_TEMPLATE_MPX.")
    # Re-derive the window from the rounded size so its centre is exactly the
    # template's centre pixel -- pose_crop measures offsets from there.
    wx1, wy1 = wx0 + RW * tpl_px, wy0 + RH * tpl_px
    x_win = 0.5 * (wx0 + wx1)
    y_win = 0.5 * (wy0 + wy1)
    if progress:
        progress(f"[frame-library] content x[{content[0]:.3f},{content[1]:.3f}] "
                 f"y[{content[2]:.3f},{content[3]:.3f}] mm")
        progress(f"[frame-library] window {wx1-wx0:.3f} x {wy1-wy0:.3f} mm "
                 f"centred ({x_win:.3f},{y_win:.3f}) -> {RW}x{RH} px "
                 f"({RW*RH/1e6:.2f} Mpx, supersample {supersample}x)")

    cam["width"], cam["height"], cam["pixel_size"] = RW, RH, tpl_px

    angles = [round(i * step_deg, 6) for i in range(int(round(360.0 / step_deg)))]
    frames = []
    t_start = time.time()
    baseline = []
    slow_run = 0
    spill_warned = False
    for i, ang in enumerate(angles):
        gono = Goniometer(scene.geometry)
        # The window offset is applied as a sample translation: tx is parallel
        # to the spindle axis so it is rotation-invariant, and ty must be
        # counter-rotated because the stage rides on the spindle.
        th = math.radians(ang)
        gono.set(**{axis: ang, "tx": -x_win,
                    "ty": -y_win * math.cos(th), "tz": y_win * math.sin(th)})
        t_frame = time.time()
        try:
            img = render_torch(tscene, gono, n_cond=n_cond, tile_size=tile_size,
                               vram_fraction=vram_fraction)
        except torch.OutOfMemoryError as exc:
            raise RuntimeError(
                f"Out of memory rendering {scene_path} at {RW}x{RH}, n_cond={n_cond}. "
                f"Lower --supersample or --vram-fraction, or pass an explicit "
                f"--tile-size."
            ) from exc
        # No OOM to catch under WSL2 -- the driver spills to host RAM and the
        # render just crawls, so timing is the only available signal. Baseline
        # off a median of early frames, NOT frame 0: the first frame carries
        # CUDA context setup, first-touch allocation and any tile calibration,
        # so it is the slowest and comparing against it never fires.
        dt_frame = time.time() - t_frame
        if len(baseline) < 8:
            baseline.append(dt_frame)
        else:
            typical = sorted(baseline)[len(baseline) // 2]
            if dt_frame > 2.5 * typical:
                slow_run += 1                 # a spill is sustained; one slow
                if slow_run >= 3 and not spill_warned and progress:
                    spill_warned = True       # frame is just a hiccup
                    progress(f"    WARNING {slow_run} consecutive frames near "
                             f"{dt_frame:.1f}s against a typical {typical:.1f}s "
                             f"-- suspect VRAM spill to host RAM; lower "
                             f"--vram-fraction")
            else:
                slow_run = 0

        arr = (img * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
        name = f"rot_{i:04d}.jpg"
        Image.fromarray(arr, mode="RGB").save(
            os.path.join(lib_dir, name), format="JPEG", quality=quality)
        frames.append({"index": i, "angle_deg": ang, "file": name})
        if progress and (i % 20 == 0 or i == len(angles) - 1):
            el = time.time() - t_start
            progress(f"    {i+1}/{len(angles)} frames  {el:6.1f}s elapsed "
                     f"({el/(i+1):.2f}s/frame)")

    manifest = {
        "scene": scene_path,
        "scene_sha256": scene_fingerprint(scene_path),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": axis,
        "step_deg": step_deg,
        "n_cond": n_cond,
        "supersample": supersample,
        "pan_mm": pan_mm,
        "jpeg_quality": quality,
        "camera": {"width": W, "height": H, "pixel_size": px0,
                   "na_objective": na_obj, "na_condenser": na_cond},
        "rendered": {"width": RW, "height": RH, "pixel_size": tpl_px},
        "window_mm": {"x0": wx0, "x1": wx1, "y0": wy0, "y1": wy1,
                      "centre_x": x_win, "centre_y": y_win},
        "frames": frames,
    }
    _write_manifest(lib_dir, manifest)
    return manifest


def ensure_library(scene_path, root=DEFAULT_ROOT, progress=print, **kwargs):
    """Return the manifest, building the library first if it is absent or stale."""
    lib_dir = library_dir(scene_path, root)
    # Resolve against build_library's own defaults, or a default build would
    # compare its manifest against None and always look stale.
    # A None reaching build_library would blow up on px0 / supersample, so drop
    # them here rather than forwarding a value this function already resolved.
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    params = build_params(**kwargs)

    if is_current(scene_path, lib_dir, **params):
        return load_manifest(lib_dir)
    if progress:
        progress(f"[frame-library] no current library for {scene_path} -- building")
    return build_library(scene_path, root=root, progress=progress, **kwargs)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def zoom_limits(manifest):
    """(min, max) zoom this library can serve at the home pose, undistorted.

    The ceiling is the supersample factor -- past it the master has no detail
    left to magnify.

    The floor is where the field of view stops fitting inside the rendered
    window.  It is NOT simply `camera / template`: the window is deliberately
    off-centre (it is anchored on the sample, which is long and thin), so at
    the home pose the camera sits `centre_x` from the template's middle and
    runs out of room on the near side first.  Zooming out past this point
    cannot show more -- the pixels do not exist -- and squeezing the crop to
    fit would change magnification per axis, so also the aspect ratio.
    """
    cam, rnd = manifest["camera"], manifest["rendered"]
    win = manifest["window_mm"]
    S = float(manifest["supersample"])
    W, H = int(cam["width"]), int(cam["height"])
    RW, RH = int(rnd["width"]), int(rnd["height"])
    tpl_px = float(rnd["pixel_size"])

    # Where the camera centre sits in the template at tx = ty = tz = 0.
    cx = RW / 2.0 - win["centre_x"] / tpl_px
    cy = RH / 2.0 - win["centre_y"] / tpl_px
    # Usable span is twice the distance to the NEARER edge, less one pixel for
    # the resampler's half-pixel edge offset.
    avail_w = max(2.0 * min(cx, RW - cx) - 1.0, 1.0)
    avail_h = max(2.0 * min(cy, RH - cy) - 1.0, 1.0)
    return max(W * S / avail_w, H * S / avail_h), S


def frame_for_angle(manifest, angle_deg):
    """Nearest pre-rendered frame record for an arbitrary spindle angle."""
    step = manifest["step_deg"]
    idx = int(round((angle_deg % 360.0) / step)) % len(manifest["frames"])
    return manifest["frames"][idx]


def pose_crop(manifest, tx=0.0, ty=0.0, tz=0.0, angle_deg=0.0, zoom=1.0,
              clamp=False):
    """Crop box, output size and defocus blur for a full goniometer pose.

    Returns `(box, out_size, sigma_px, note)`:

      box       float source box for `Image.resize(out_size, box=box)` -- NOT an
                integer crop box. Kept in floats so the resampler samples the
                template on exactly the camera's grid; rounding it to integers
                costs a sub-pixel registration error and makes the magnification
                flicker by ~0.1% as a pan crosses pixel boundaries.
      out_size  the camera resolution to resize to
      sigma_px  Gaussian blur approximating condenser defocus
      note      None, or a description of what had to be clamped

    `clamp=False` (the default, for programmatic use) raises when the pose falls
    outside what the library can serve.  `clamp=True` brings it back in range and
    returns `note` saying so -- the live server must keep serving, but a silently
    clamped pan is indistinguishable from a correct one once it is on screen, so
    the caller is expected to surface `note`.

    The XYZ stage rides on the spindle (Goniometer composes T = Rz.Ry.Rx.T),
    so the lab displacement is R*(tx,ty,tz), not (tx,ty,tz).  With the spindle
    on rotx that puts `ty*cos - tz*sin` in the image plane and
    `ty*sin + tz*cos` along the view axis, which is why a plain ty crop is
    wrong at any phi != 0 and a no-op at phi = 90.
    """
    cam, rnd = manifest["camera"], manifest["rendered"]
    win = manifest["window_mm"]
    S = manifest["supersample"]
    W, H = int(cam["width"]), int(cam["height"])
    tpl_px = float(rnd["pixel_size"])
    RWp, RHp = int(rnd["width"]), int(rnd["height"])
    note = None

    zmin, zmax = zoom_limits(manifest)
    if zoom <= 0:
        raise ValueError(f"zoom must be positive, got {zoom}")
    if zoom > zmax + 1e-9 or zoom < zmin - 1e-9:
        msg = (f"zoom {zoom:g} is outside the {zmin:.3f}-{zmax:g}x this library "
               f"can serve: above {zmax:g} the master has no detail left to "
               f"magnify (rebuild with a larger --supersample); below "
               f"{zmin:.3f} the field of view is wider than the rendered window "
               f"(rebuild with a larger --pan-mm)")
        if not clamp:
            raise ValueError(msg)
        # Clamp zoom rather than the box: clamping the box would squeeze the
        # two axes independently and silently change the aspect ratio.
        zoom = min(max(zoom, zmin), zmax)
        note = msg

    # The sweep was rendered at a motor offset of (-centre_x, -centre_y), so the
    # image shift from template to requested pose is R*(t_req - t_tpl).
    th = math.radians(angle_deg)
    u = tx + win["centre_x"]
    v = ty * math.cos(th) - tz * math.sin(th) + win["centre_y"]
    w = ty * math.sin(th) + tz * math.cos(th)

    # Template pixels per output pixel.
    scale = S / float(zoom)
    span_w, span_h = W * scale, H * scale
    cx = RWp / 2.0 - u / tpl_px
    cy = RHp / 2.0 - v / tpl_px

    # PIL maps output pixel i to source EDGE coord left + (i+0.5)*scale, i.e.
    # source index left + i*scale + 0.5*(scale-1). The renderer samples camera
    # pixel i at index cx + (i - W/2)*scale, so the edge origin is offset by
    # half a source pixel. Dropping this costs 0.375 template px at 4x.
    left = cx - span_w / 2.0 - 0.5 * scale + 0.5
    upper = cy - span_h / 2.0 - 0.5 * scale + 0.5
    right, lower = left + span_w, upper + span_h

    # Slide the box inside the template. Two different things can push it out:
    # a genuine pan past the rendered window, and -- at the extreme zoom-out
    # where the span exactly fills the template -- the half-pixel edge offset
    # above. Only the first is worth telling anyone about, so the excursion is
    # measured and a sub-pixel one is absorbed silently.
    #
    # Sliding never squeezes: span_w/span_h are preserved, so magnification and
    # aspect ratio are untouched. The zoom guard guarantees the span fits, so a
    # pure translation always suffices.
    slid_x = min(max(left, 0.0), max(RWp - span_w, 0.0)) - left
    slid_y = min(max(upper, 0.0), max(RHp - span_h, 0.0)) - upper
    if abs(slid_x) > 1.0 or abs(slid_y) > 1.0:
        pan_msg = (f"pose (tx={tx:g}, ty={ty:g}, tz={tz:g}, phi={angle_deg:g}, "
                   f"zoom={zoom:g}) needs template pixels "
                   f"[{left:.0f},{upper:.0f}]-[{right:.0f},{lower:.0f}] outside "
                   f"the rendered {RWp}x{RHp} window -- rebuild with a larger "
                   f"--pan-mm, or zoom in")
        if not clamp:
            raise ValueError(pan_msg)
        note = pan_msg if note is None else note + "; " + pan_msg
    left, upper = left + slid_x, upper + slid_y
    right, lower = left + span_w, upper + span_h

    eff_px = float(cam["pixel_size"]) / zoom
    sigma_px = 0.5 * float(cam["na_condenser"]) * abs(w) / eff_px
    return (left, upper, right, lower), (W, H), sigma_px, note
