"""Pre-computed frame library: a rotation sweep rendered once and replayed.

The camera is orthographic, so the spindle is the only motor that changes
image content. A library is a directory of PNG (or JPEG) frames around a measured
render window, one per spindle angle, plus a manifest.json recording the
scene's SHA-256, the build parameters, and each frame's file name and crop
offset/size. `library_status` reads a library as `current` (matches the
requested build parameters), `stale` (complete and servable, built with
different ones -- see `library_diff` and `describe_differences`) or
`missing` (unusable).  Only `missing` ever provokes a build: a stale
library is served as it stands, because rebuilding one is hours of GPU time
and `python -m loop_sim.library --force` is the only thing entitled to
spend them.  `library_provenance` says when a library was built and from
what, and `verify_frame` re-renders one stored frame to measure how far the
renderer has moved since.  `pose_crop`/`servable_pose` turn a goniometer
pose into the crop, size and blur to serve; `frame_for_angle` and
`zoom_limits` pick a sweep frame and bound the servable zoom.
"""
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import time

import numpy as np

# Anchored to the repo root rather than the caller's cwd: a library written to
# the wrong directory is a silently missing deliverable, since it has to land in git.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ROOT = os.path.join(_REPO_ROOT, "data", "frame_library")
# Coarse stand-in libraries, built on demand when someone switches the live
# server to a scene that has none.  They go in a SEPARATE root because building
# into the live one would overwrite frames the serving TemplateSource is
# decoding and caching by filename -- the reader would keep serving whichever
# mixture of old and new bytes its cache happened to hold.  Untracked, unlike
# data/frame_library/: these are disposable, not a deliverable.
DEFAULT_PREVIEW_ROOT = DEFAULT_ROOT + "_preview"
# ~72 frames instead of 360, at the camera's own pitch and a single condenser
# ray: minutes rather than the best part of an hour, at the cost of 5deg
# rotation granularity and a zoom ceiling of 1x.
PREVIEW_BUILD = {"step_deg": 5.0, "supersample": 1, "n_cond": 1}
DEFAULT_STEP_DEG = 1.0
DEFAULT_SUPERSAMPLE = 4
DEFAULT_PAN_MM = 0.6
DEFAULT_N_COND = 7
DEFAULT_QUALITY = 90
# Lossless: a real AXIS camera applies exactly one JPEG compression, and a
# stored JPEG re-encoded on the wire would apply a second, a signature no
# real camera has. PNG is also smaller for these near-binary frames, at the
# cost of decode speed. See docs/DECISIONS.md 2026-08-06 (realism pass: the
# objective PSF, and lossless templates).
DEFAULT_FORMAT = "png"
# PNG compression level.  NOT a build parameter: it changes file size, never a
# pixel, so it must not invalidate a library.
PNG_COMPRESS_LEVEL = 6
# Convolve each template with the objective PSF as it is rendered (see
# renderer/optics.py).  Without it the templates are geometrically sharper than
# the optics can form, which shows as blocky edges once you magnify.
DEFAULT_PSF = True
DEFAULT_VRAM_FRACTION = 0.80
# Sanity guard on the measured render window, not a hardware limit: 4x
# supersample over a 10 x 5 mm window is ~14 Mpx, so anything past this means
# the content measurement went wrong.
MAX_TEMPLATE_MPX = 200.0

# Only the content is stored, not the full render window; the reader fills
# the rest. This is exact, not approximate: every ray is born at radiance
# 1.0, so anything the sample does not touch is exactly white -- recorded
# in the manifest rather than assumed, so the format says what to fill with.
# See docs/DECISIONS.md 2026-08-14 (templates store content only).
BACKGROUND_RGB = (255, 255, 255)


def crop_margin_px(supersample, zoom_min):
    """Template pixels to keep around the content, so the reader never clips it.

    Two things eat into the margin at the edge of the stored crop, and both
    scale with `scale = supersample / zoom` -- the template pixels per output
    pixel, maximised at the lowest servable zoom:

      * the reader rounds its output sub-rect INWARD (rounding outward would
        need source pixels that were never stored, and PIL refuses a negative
        box offset), discarding up to one output pixel = `scale` template px;
      * PIL's bilinear support at that downscale reaches about `scale` px past
        the sample point.

    So `2 * scale`, plus a few pixels for the objective PSF's spread, and a
    floor because the arithmetic is only worth trusting when it is generous.
    For the shipped hampton sweep (supersample 4, zoom floor 0.798) it comes to
    the floor of 16, which is the value the crop was validated at.
    """
    want = 2.0 * float(supersample) / max(float(zoom_min), 1e-6) + 4.0
    return max(16, int(math.ceil(want)))


def content_bbox(arr, background=BACKGROUND_RGB):
    """`(x0, y0, x1, y1)` inclusive bbox of everything that is not background.

    `None` when the frame is entirely background -- a real possibility for a
    scene whose sample leaves the field at some angles, and the caller must not
    crash on it.

    Exact rather than thresholded: the whole argument for cropping is that it
    is lossless, and a tolerance would quietly make it not.  Measured on the
    shipped libraries, exact costs 2 px per side against a 2/255 threshold.
    """
    bg = np.asarray(background, dtype=arr.dtype)
    if np.all(bg == bg[0]):
        mask = arr.min(axis=2) < int(bg[0])       # fast path: uniform background
    else:
        mask = (arr != bg).any(axis=2)
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]), int(rows[-1])


def crop_to_content(arr, margin, background=BACKGROUND_RGB):
    """`(cropped, (origin_x, origin_y))` -- the content plus `margin`, clamped.

    A blank frame yields a 1x1 crop at the frame centre rather than an empty
    array: PIL cannot store a zero-size image, and the reader treats "the box
    misses the stored crop entirely" as all-background anyway, so the degenerate
    case costs one pixel and needs no special case downstream.
    """
    h, w = arr.shape[:2]
    box = content_bbox(arr, background)
    if box is None:
        cx, cy = w // 2, h // 2
        return arr[cy:cy + 1, cx:cx + 1].copy(), (cx, cy)
    x0, y0, x1, y1 = box
    ox = max(x0 - margin, 0)
    oy = max(y0 - margin, 0)
    ex = min(x1 + 1 + margin, w)
    ey = min(y1 + 1 + margin, h)
    return arr[oy:ey, ox:ex].copy(), (ox, oy)

# Build parameters that change the pixels. A library whose manifest disagrees
# with the requested value of any of these is stale, not merely different.
_BUILD_KEYS = ("axis", "step_deg", "n_cond", "supersample", "pan_mm",
               "jpeg_quality", "format", "psf")

# Modules whose SOURCE decides what a template pixel is.  ADVISORY ONLY: the
# digest over them is stamped into the manifest and reported by
# `library_provenance`, and `verify_frame` measures what actually changed, but
# nothing here grades a library stale.  An edit to a tracer covers every
# library at once, and a verdict that costs hours to clear has to be earned by
# a measurement rather than by a hash.  Deliberately NOT hashed at all:
#   renderer/field.py   camera emulation, applied at SERVE time downstream of
#                       `pose_crop`, never baked into a template -- see
#                       docs/DECISIONS.md 2026-08-10 (the renders became
#                       photographs).  Living outside the tracer is what lets
#                       it change without forcing a rebuild.
#   renderer/beam.py, renderer/xray_torch.py   the X-ray path, no optical
#                       template comes from either.  Keep X-ray code out of
#                       this file -- see docs/DECISIONS.md 2026-08-18
#                       (xray_torch split out of engine_torch.py).
#   library/, server/   delivery, not content -- and `pose_crop` lives in
#                       library/, so hashing it would invalidate every library
#                       for a change to how frames are CROPPED.
_RENDER_SOURCES = ("renderer/microscope.py", "renderer/engine_torch.py",
                   "renderer/optics.py", "motors/goniometer.py", "scene/*.py")
_render_sha_cache = None


def scene_fingerprint(scene_path):
    with open(scene_path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def render_source_paths():
    """`(package_root, sorted_paths)` -- exactly the files `render_sha` hashes.

    Public so a test can assert WHICH files are covered.  The set still has to
    be right even though it no longer invalidates anything: it is what
    `library_provenance` advises on and what `--verify` is answering a question
    about, and a tracer left out of it never raises a flag at all.
    """
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = []
    for pat in _RENDER_SOURCES:
        paths.extend(glob.glob(os.path.join(pkg, *pat.split("/"))))
    return pkg, sorted(paths)


def _sha_over(root, paths):
    """sha256 of (relative path, contents) for each file, in the given order.

    The PATH is hashed alongside the contents so that adding, removing or
    renaming a module registers -- a bare content digest would not notice a
    file that had been deleted.
    """
    h = hashlib.sha256()
    for path in paths:
        h.update(os.path.relpath(path, root).replace(os.sep, "/").encode())
        with open(path, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()


def render_sha():
    """sha256 over the source of every module that decides a template pixel.

    Computed once per process: these files cannot change under a running build,
    and reading a dozen of them every time a provenance line is printed would
    make listing scenes on the control page do pointless I/O.
    """
    global _render_sha_cache
    if _render_sha_cache is None:
        _render_sha_cache = _sha_over(*render_source_paths())
    return _render_sha_cache


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

    The manifest is the one file that makes the other 360 usable, so an
    unflushed write a host crash loses orphans the whole library. Temp-file
    + fsync + rename also means a crash mid-write can never leave a
    truncated manifest behind.
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
                 quality=DEFAULT_QUALITY, format=DEFAULT_FORMAT,
                 psf=DEFAULT_PSF, **_ignored):
    """The pixel-affecting settings, keyed as the manifest stores them.

    One place to resolve defaults, so `library_status` cannot drift from
    `build_library` and start reporting every default build as stale (or, worse,
    every changed build as current).

    Every value here must resolve to something concrete -- `library_diff` skips
    any key whose requested value is None, so a None default would silently
    disable staleness checking for that parameter.

    Every key is a build SETTING, and a caller declaring one declares only what
    it is asking for, never that a library already satisfies it.  The state of
    the renderer's own source is not settled here: it is not a setting, it is
    not something a caller gets to assert, and a digest that graded libraries
    made one edit to a tracer cost a rebuild of all of them.  `render_sha` is
    still stamped into the manifest, and `library_provenance` and `verify_frame`
    are what answer for it.
    """
    return {"axis": axis, "step_deg": step_deg, "n_cond": n_cond,
            "supersample": supersample, "pan_mm": pan_mm,
            "jpeg_quality": quality, "format": format, "psf": bool(psf)}


def _frames_complete(lib_dir, man):
    """True when every frame the manifest lists is on disk and undamaged.

    The completeness half of `library_status`: a library that fails this reads
    as 'missing', never as 'stale' (servable), so a half-written one is refused.
    """
    if not man.get("frames"):
        return False
    if not all(os.path.exists(os.path.join(lib_dir, f["file"]))
               for f in man["frames"]):
        return False
    # Cheap guard against a half-overwritten library: check one frame really is
    # the size the manifest claims. pose_crop computes boxes from the manifest,
    # and PIL pads an over-large box with black rather than raising, so a size
    # mismatch would serve silently corrupt frames.
    #
    # This is ALSO the interlock between cropped and uncropped libraries, and it
    # is the only one that bites: a stale library still serves (the server warns
    # and carries on), but "missing" refuses outright.  Comparing against the
    # DECLARED stored size makes it symmetric and free -- cropped frames read by
    # code that ignores `content_size_px` fail the check, and a manifest
    # claiming a crop over full-window frames fails it too.  Absence of the
    # field means "the stored image IS the virtual window", which is what keeps
    # every pre-crop library working untouched.
    try:
        from PIL import Image
        rec = man["frames"][0]
        want = rec.get("content_size_px") or (man["rendered"]["width"],
                                              man["rendered"]["height"])
        with Image.open(os.path.join(lib_dir, rec["file"])) as im:
            if im.size != (int(want[0]), int(want[1])):
                return False
    except Exception:
        return False
    return True


def _stored_format(man):
    """The format a manifest's templates are actually stored in.

    A library built before `format` became a build key does not record it, so
    the file extension is the ground truth -- and it is also what the operator
    sees on disk.  Reporting "unset" instead would make the staleness warning
    useless precisely for the libraries that need it.
    """
    fmt = man.get("format")
    if fmt:
        return fmt
    ext = os.path.splitext(man["frames"][0]["file"])[1].lower()
    return "jpeg" if ext in (".jpg", ".jpeg") else "png"


def library_diff(scene_path, lib_dir, **params):
    """Which BUILD PARAMETERS a library on disk disagrees with `params` about.

    Returns {key: {"have":…, "want":…}}, empty when they match.  `library_status`
    only answers current/stale/missing; anything that serves a stale library
    anyway has to tell the operator WHAT differs, and re-deriving that in the
    caller is how the two would drift apart.

    A key absent from the manifest reads as None and so differs from any
    concrete request -- exactly `data/frame_library/mitegen_200um`, written before
    `format` and `psf` existed.  The scene fingerprint is deliberately NOT
    considered here: a changed scene means the frames show a different object,
    which is `library_status`'s "missing", not a difference of degree.
    """
    man = load_manifest(lib_dir)
    if man is None or not man.get("frames"):
        return {}
    out = {}
    for key in _BUILD_KEYS:
        want = params.get(key)
        if want is None:
            continue                    # not asked about -- not a difference
        have = _stored_format(man) if key == "format" else man.get(key)
        if have != want:
            out[key] = {"have": have, "want": want}
    return out


def library_status(scene_path, lib_dir, **params):
    """'current' | 'stale' | 'missing' for a library on disk.

    Three answers rather than a bool, because the two failure modes need
    opposite handling:

      missing -- nothing usable: no manifest, frames absent or damaged, or the
                scene YAML has changed since the build (those frames are of a
                different object, so serving them would be a lie).  The only
                status that provokes a build.
      stale   -- a COMPLETE, servable library that simply was not built with
                the parameters being asked for.  `data/frame_library/mitegen_200um`
                is exactly this: 360 usable frames, ~1.9 h to reproduce, whose
                only sin is a manifest older than the `format` and `psf` build
                keys.  Refusing to serve it -- or silently rebuilding -- would
                cost far more than the warning it deserves.  `library_diff`
                says what differs and `describe_differences` puts it in
                English.
    """
    man = load_manifest(lib_dir)
    if man is None:
        return "missing"
    if man.get("scene_sha256") != scene_fingerprint(scene_path):
        return "missing"
    if not _frames_complete(lib_dir, man):
        return "missing"
    return "stale" if library_diff(scene_path, lib_dir, **params) else "current"


_DIFF_PHRASES = {
    "format":       lambda d: f"stored as {d['have']}, not {d['want']}",
    "psf":          lambda d: ("built without the objective PSF" if not d["have"]
                               else "built with the objective PSF"),
    "supersample":  lambda d: (f"{d['have']}x supersample, so zoom is capped at "
                               f"{d['have']}x rather than {d['want']}x"),
    "step_deg":     lambda d: f"{d['have']:g}deg rotation steps, not {d['want']:g}",
    "n_cond":       lambda d: f"{d['have']} condenser rays, not {d['want']}",
    "pan_mm":       lambda d: f"{d['have']} mm pan margin, not {d['want']}",
    "jpeg_quality": lambda d: f"JPEG quality {d['have']}, not {d['want']}",
    "axis":         lambda d: f"swept about {d['have']}, not {d['want']}",
}


def describe_differences(differs):
    """One operator-readable sentence naming what a stale library differs in.

    "stale" on its own tells nobody whether to care.  Lives here rather than in
    the server so the CLI, the log line, the tab tooltip and the control-page
    banner cannot disagree.
    """
    if not differs:
        return None
    parts = [_DIFF_PHRASES.get(k, lambda d, k=k: f"{k} is {d['have']}, not {d['want']}")(v)
             for k, v in sorted(differs.items())]
    return ("this frame library was built with different settings: "
            + "; ".join(parts)
            + ". It is complete and is being served as-is -- rebuild it only if "
              "you need those settings.")


def library_provenance(manifest):
    """One line: when a library was built, from what commit, and how.

    The whole of what the manifest's advisory fields are for.  `render_sha` no
    longer grades anything, so this is the only place an operator learns that
    the tracer has moved on since the build -- and it says what to do about it
    rather than declaring hours of rebuild owed.

    Never raises.  It goes in a launch banner and a CLI listing, and a manifest
    missing a field must not take either down.
    """
    if not isinstance(manifest, dict):
        return "no manifest"
    try:
        stamp = str(manifest.get("built_utc") or "")
        # "2026-08-11T21:30:05Z" -> "2026-08-11 21:30 UTC"; anything else is
        # printed as it was stored rather than guessed at.
        if len(stamp) >= 16 and stamp[10] in "T ":
            parts = [f"built {stamp[:10]} {stamp[11:16]} UTC"]
        elif stamp:
            parts = [f"built {stamp}"]
        else:
            parts = ["build date unknown"]
        if manifest.get("built_commit"):
            parts.append(f"commit {manifest['built_commit']}")
        for label, key in (("supersample", "supersample"), ("n_cond", "n_cond")):
            if manifest.get(key) is not None:
                parts.append(f"{label} {manifest[key]}")
        if manifest.get("step_deg") is not None:
            parts.append(f"step {manifest['step_deg']} deg")
        fmt = manifest.get("format")
        if not fmt and manifest.get("frames"):
            fmt = _stored_format(manifest)
        if fmt:
            parts.append(str(fmt))
        line = ", ".join(parts)
        sha = manifest.get("render_sha")
        if sha and sha != render_sha():
            line += ("; renderer source changed since the build "
                     "(run --verify or rebuild)")
        return line
    except Exception:
        return "provenance unavailable"


def _built_commit():
    """`git describe --always --dirty` in the repo root, or "unknown".

    Provenance, not a gate: a library whose build cannot be tied to a commit is
    still perfectly servable, so every way this can fail -- no git, no repo, a
    hung index lock -- reads as "unknown" rather than losing a build that has
    already run for an hour.
    """
    try:
        out = subprocess.run(["git", "describe", "--always", "--dirty"],
                             cwd=_REPO_ROOT, capture_output=True, text=True,
                             timeout=10)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


CPU_BUILD_REFUSAL = (
    "no CUDA device -- refusing to build a frame library. On CPU this renders "
    "at roughly 179 s/frame: a 72-frame preview is ~3.6 h and a 360-frame "
    "library is ~18 h. Build it on a GPU host with `python -m loop_sim.library "
    "--scene <scene>`, then switch to it; pass --allow-cpu to that command if "
    "you really do mean to build on CPU.")


def cuda_available():
    """True when a build would actually run on a GPU.

    One definition, so the server and the CLI refuse on identical grounds, and
    one place for a test to monkeypatch.  A torch import failure reads as "no
    CUDA" rather than propagating: the answer to "can we build fast?" is no
    either way.
    """
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


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
    from ..renderer.torch_compat import ensure_dynamo
    ensure_dynamo()
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
def render_sweep_frame(scene, tscene, angle_deg, x_win, y_win, axis="rotx",
                       n_cond=DEFAULT_N_COND, psf=DEFAULT_PSF, tile_size="fit",
                       vram_fraction=DEFAULT_VRAM_FRACTION):
    """One sweep frame, as a uint8 HxWx3 array, at the pose the sweep uses.

    The window offset is applied as a sample translation: tx is parallel to the
    spindle axis so it is rotation-invariant, and ty must be counter-rotated
    because the stage rides on the spindle.

    `verify_frame` renders through here too, so there is one copy of that
    arithmetic: a verification that re-derived the pose would be testing its own
    derivation as much as the renderer.  The caller sets the camera to the
    template geometry first -- `render_torch` takes width/height/pixel_size off
    the scene, never from arguments.
    """
    import torch
    from ..motors.goniometer import Goniometer
    from ..renderer.engine_torch import render_torch

    th = math.radians(angle_deg)
    gono = Goniometer(scene.geometry).set(
        **{axis: angle_deg, "tx": -x_win,
           "ty": -y_win * math.cos(th), "tz": y_win * math.sin(th)})
    img = render_torch(tscene, gono, n_cond=n_cond, psf=psf,
                       tile_size=tile_size, vram_fraction=vram_fraction)
    return (img * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()


def build_library(scene_path, root=DEFAULT_ROOT, axis="rotx",
                  step_deg=DEFAULT_STEP_DEG, supersample=DEFAULT_SUPERSAMPLE,
                  pan_mm=DEFAULT_PAN_MM, n_cond=DEFAULT_N_COND,
                  quality=DEFAULT_QUALITY, format=DEFAULT_FORMAT,
                  psf=DEFAULT_PSF, tile_size=None,
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
    from ..renderer.torch_compat import ensure_dynamo
    ensure_dynamo()   # torch 2.0.1 does not bind torch._dynamo itself
    from ..renderer.engine_torch import TorchScene, check_render_fits
    from ..renderer.optics import psf_sigma_px

    if axis != "rotx":
        # build_library's window offset and pose_crop both hard-code the rotx
        # coupling (tx parallel to the spindle, ty/tz counter-rotated). Another
        # axis needs both generalised; failing loudly beats a wrong library.
        raise ValueError(
            f"axis={axis!r} is not supported yet -- the window offset and "
            f"pose_crop both assume the spindle is rotx. Generalise both "
            f"before enabling another axis.")

    fmt = str(format).lower()
    if fmt not in ("png", "jpeg"):
        raise ValueError(f"format={format!r} must be 'png' or 'jpeg'")
    ext = "png" if fmt == "png" else "jpg"

    lib_dir = library_dir(scene_path, root)
    # Render into a sibling directory and swap at the end.  Until the swap
    # nothing on disk is touched, so an OOM at frame 300 of 360, a Ctrl-C or a
    # dead host all leave the library that was already there exactly as it was.
    # It also settles what building in place needed two pieces of bookkeeping
    # for: a manifest describing a mixture of old and new frames, and a format
    # change orphaning the previous extension's images inside a tracked
    # directory.
    new_dir = lib_dir + ".new"
    old_dir = lib_dir + ".old"
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir)

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
    # Recorded in the manifest for provenance: how much optical softening was
    # baked in, in template pixels.
    psf_sigma = psf_sigma_px(cam, tpl_px) if psf else 0.0
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

    # PREFLIGHT: render one real frame and check it fits, rather than
    # discovering the memory ceiling at frame 300 of 360.  Shrinks the trace
    # tile if that's enough (byte-exact, so the library is unaffected), or
    # raises RenderTooLargeError naming the largest --supersample that would
    # fit.  See docs/DECISIONS.md 2026-08-11 (the VRAM budget is enforced,
    # not assumed).
    if tile_size is None:
        tile_size = check_render_fits(tscene, n_cond=n_cond, psf=psf,
                                      vram_fraction=vram_fraction,
                                      supersample=supersample, progress=progress)

    # Everything `zoom_limits` and `pose_crop` need, assembled before the sweep
    # so the crop margin is derived from the SAME geometry the reader will use
    # rather than from a second copy of the arithmetic.
    geom = {"camera": {"width": W, "height": H, "pixel_size": px0},
            "rendered": {"width": RW, "height": RH, "pixel_size": tpl_px},
            "window_mm": {"centre_x": x_win, "centre_y": y_win},
            "supersample": supersample}
    margin = crop_margin_px(supersample, zoom_limits(geom)[0])

    angles = [round(i * step_deg, 6) for i in range(int(round(360.0 / step_deg)))]
    # Report ~20 times whatever the frame count, rather than every 20th frame:
    # a coarse 72-frame preview built for the live scene switcher would
    # otherwise emit 5 lines, and the server's progress bar is only as good as
    # this.  A 360-frame build reports 20 times, as it already did.
    prog_every = max(1, len(angles) // 20)
    frames = []
    t_start = time.time()
    baseline = []
    slow_run = 0
    spill_warned = False
    try:
        for i, ang in enumerate(angles):
            t_frame = time.time()
            try:
                arr = render_sweep_frame(scene, tscene, ang, x_win, y_win,
                                         axis=axis, n_cond=n_cond, psf=psf,
                                         tile_size=tile_size,
                                         vram_fraction=vram_fraction)
            except torch.OutOfMemoryError as exc:
                raise RuntimeError(
                    f"Out of memory rendering {scene_path} at {RW}x{RH}, n_cond={n_cond}. "
                    f"Lower --supersample or --vram-fraction, or pass an explicit "
                    f"--tile-size."
                ) from exc
            # No OOM to catch under WSL2 -- the driver spills to host RAM and
            # the render just crawls, so timing is the only available signal.
            # Baseline off a median of early frames, NOT frame 0: the first
            # frame carries CUDA context setup, first-touch allocation and any
            # tile calibration, so it is the slowest and comparing against it
            # never fires.
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

            # Cropped from the frame's own pixels, not `content_window`'s coarse
            # scout (320x240, n_cond=1, effectively no PSF), which under-measures
            # and would clip real sample here.  Per frame, not one box for the
            # sweep: `mitegen_200um` needs 231 distinct bboxes across its 360
            # angles, against 1 for hampton.  See docs/DECISIONS.md 2026-08-14
            # (templates store content only).
            arr, (ox, oy) = crop_to_content(arr, margin)
            name = f"rot_{i:04d}.{ext}"
            if fmt == "png":
                Image.fromarray(arr, mode="RGB").save(
                    os.path.join(new_dir, name), format="PNG",
                    compress_level=PNG_COMPRESS_LEVEL)
            else:
                Image.fromarray(arr, mode="RGB").save(
                    os.path.join(new_dir, name), format="JPEG", quality=quality)
            frames.append({"index": i, "angle_deg": ang, "file": name,
                           "content_origin_px": [ox, oy],
                           "content_size_px": [arr.shape[1], arr.shape[0]]})
            if progress and (i % prog_every == 0 or i == len(angles) - 1):
                el = time.time() - t_start
                progress(f"    {i+1}/{len(angles)} frames  {el:6.1f}s elapsed "
                         f"({el/(i+1):.2f}s/frame)")
    except BaseException:
        # BaseException, not Exception: Ctrl-C during a multi-hour build is the
        # likeliest way this ends, and it must report the same thing an OOM
        # does.  The partial build is left where it is rather than cleaned up --
        # a half-finished sweep is the only evidence of what went wrong.
        if progress:
            progress(f"[frame-library] build did not finish -- {lib_dir} is "
                     f"untouched; the partial build is at {new_dir}")
        raise

    manifest = {
        "scene": scene_path,
        "scene_sha256": scene_fingerprint(scene_path),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "built_commit": _built_commit(),
        "axis": axis,
        "step_deg": step_deg,
        "n_cond": n_cond,
        "supersample": supersample,
        "pan_mm": pan_mm,
        "jpeg_quality": quality,
        "format": fmt,
        "psf": bool(psf),
        "render_sha": render_sha(),
        # Provenance only -- not a build key. The sigma is derived from the
        # camera NA and the rendered pixel size, both recorded below.
        "psf_sigma_px": psf_sigma,
        "camera": {"width": W, "height": H, "pixel_size": px0,
                   "na_objective": na_obj, "na_condenser": na_cond},
        # THE VIRTUAL WINDOW, not the stored image size.  `pose_crop`,
        # `zoom_limits`, `servable_pose` and `pin_projection.template_mapper`
        # all key off this, so keeping it virtual means the served geometry is
        # bit-for-bit what it was before the crop existed and none of them had
        # to change.  Where each frame's pixels actually sit inside it is on the
        # frame record.
        "rendered": {"width": RW, "height": RH, "pixel_size": tpl_px},
        "window_mm": {"x0": wx0, "x1": wx1, "y0": wy0, "y1": wy1,
                      "centre_x": x_win, "centre_y": y_win},
        "background_rgb": list(BACKGROUND_RGB),
        "crop_margin_px": margin,
        "frames": frames,
    }
    _write_manifest(new_dir, manifest)

    # The swap.  Two renames, so the window in which `lib_dir` is not a whole
    # library is the gap between them: a reader that opens the manifest before
    # or after sees one consistent sweep either way.  `.old` goes last, because
    # deleting 360 frames is the one step here that can take a while.
    if os.path.exists(old_dir):
        shutil.rmtree(old_dir)
    if os.path.exists(lib_dir):
        os.rename(lib_dir, old_dir)
    os.rename(new_dir, lib_dir)
    shutil.rmtree(old_dir, ignore_errors=True)
    return manifest


def recrop_library(lib_dir, progress=print):
    """Crop an already-built library's frames to their content, in place.

    A migration, not a build: the pixels that survive are bit-identical to the
    ones already on disk, so this needs no GPU, no scene and no re-render, and
    it leaves `scene_sha256`, `render_sha` and every build key alone -- the
    library does not become stale by being cropped.  Minutes against the hours
    a rebuild costs, and the shipped libraries get SMALLER (28.9 -> 12.9 MB for
    the realistic hampton sweep, and 15.5 -> 1.8 GB decoded).

    Cropping the window would barely help the BUILD -- measured at 1.08x for
    8.8x fewer pixels, because the AABB cull already makes background rays
    nearly free -- so there is deliberately no path here that re-renders.

    Idempotent: a library that already carries `content_origin_px` is returned
    untouched.  The manifest is rewritten LAST, so an interrupted run leaves
    frames that disagree with a manifest still claiming the full window, which
    `_frames_complete` correctly reports as missing rather than serving.
    """
    from PIL import Image

    man = load_manifest(lib_dir)
    if man is None:
        raise FileNotFoundError(f"no manifest in {lib_dir}")
    if any("content_origin_px" in f for f in man["frames"]):
        if progress:
            progress(f"[frame-library] {lib_dir} is already cropped")
        return man

    rnd = man["rendered"]
    margin = crop_margin_px(man["supersample"], zoom_limits(man)[0])
    background = tuple(man.get("background_rgb") or BACKGROUND_RGB)
    fmt = _stored_format(man)
    t0 = time.time()
    frames, before, after = [], 0, 0

    for i, rec in enumerate(man["frames"]):
        path = os.path.join(lib_dir, rec["file"])
        before += os.path.getsize(path)
        with Image.open(path) as im:
            arr = np.asarray(im.convert("RGB"))
        if arr.shape[1::-1] != (int(rnd["width"]), int(rnd["height"])):
            raise ValueError(
                f"{rec['file']} is {arr.shape[1]}x{arr.shape[0]} but the manifest "
                f"says {rnd['width']}x{rnd['height']} -- refusing to crop a "
                f"library that is already inconsistent")
        arr, (ox, oy) = crop_to_content(arr, margin, background)
        if fmt == "png":
            Image.fromarray(arr, mode="RGB").save(
                path, format="PNG", compress_level=PNG_COMPRESS_LEVEL)
        else:
            Image.fromarray(arr, mode="RGB").save(
                path, format="JPEG", quality=man.get("jpeg_quality", DEFAULT_QUALITY))
        after += os.path.getsize(path)
        frames.append(dict(rec, content_origin_px=[ox, oy],
                           content_size_px=[arr.shape[1], arr.shape[0]]))
        if progress and (i + 1) % max(1, len(man["frames"]) // 10) == 0:
            progress(f"    {i+1}/{len(man['frames'])} frames")

    man = dict(man)
    man["frames"] = frames
    man["background_rgb"] = list(background)
    man["crop_margin_px"] = margin
    _write_manifest(lib_dir, man)
    if progress:
        px = sum(f["content_size_px"][0] * f["content_size_px"][1] for f in frames)
        full = int(rnd["width"]) * int(rnd["height"]) * len(frames)
        progress(f"[frame-library] cropped {len(frames)} frames in "
                 f"{time.time() - t0:.1f}s -- {before/1e6:.1f} -> {after/1e6:.1f} MB "
                 f"on disk, {full*3/1e9:.2f} -> {px*3/1e9:.2f} GB decoded "
                 f"({px/full:.1%} of the window, margin {margin} px)")
    return man


def verify_frame(scene_path, lib_dir, angle_deg=0.0, device=None,
                 tile_size=None, vram_fraction=DEFAULT_VRAM_FRACTION,
                 progress=None):
    """Re-render one stored frame live and measure how far it has drifted.

    Returns `(rec, max_abs, mean_abs)`: the frame record, and the worst and mean
    absolute difference in grey levels over the whole rendered window.

    This is what replaces grading on `render_sha`.  The digest could only say
    that a tracer's source had changed, never whether a pixel had, and it
    charged hours of rebuild for the difference.  One frame costs seconds and
    answers the question that was actually being asked.

    The comparison runs on the WINDOW, not on the two crops: a stored crop and
    a fresh one are the same picture only while the content bbox agrees, and a
    box that has moved is itself a difference worth measuring rather than a
    shape error.
    """
    import torch
    from PIL import Image
    from ..scene.scene import load
    from ..renderer.torch_compat import ensure_dynamo
    ensure_dynamo()   # torch 2.0.1 does not bind torch._dynamo itself
    from ..renderer.engine_torch import TorchScene, check_render_fits

    man = load_manifest(lib_dir)
    if man is None:
        raise FileNotFoundError(f"no manifest in {lib_dir}")
    rec = frame_for_angle(man, angle_deg)
    rnd, win = man["rendered"], man["window_mm"]

    scene = load(scene_path, device="cpu")
    cam = scene.camera_cfg
    # The template geometry, not the camera's: render_torch reads width, height
    # and pixel_size off the scene, so this IS how the build set the window.
    cam["width"] = int(rnd["width"])
    cam["height"] = int(rnd["height"])
    cam["pixel_size"] = float(rnd["pixel_size"])

    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    tscene = TorchScene(scene, dev, torch.float64)
    if tile_size is None:
        tile_size = check_render_fits(tscene, n_cond=man["n_cond"],
                                      psf=man.get("psf", True),
                                      vram_fraction=vram_fraction,
                                      supersample=man["supersample"],
                                      progress=progress)

    arr = render_sweep_frame(scene, tscene, rec["angle_deg"],
                             float(win["centre_x"]), float(win["centre_y"]),
                             axis=man.get("axis", "rotx"),
                             n_cond=man["n_cond"], psf=man.get("psf", True),
                             tile_size=tile_size, vram_fraction=vram_fraction)

    stored = np.empty_like(arr)
    stored[:, :] = np.asarray(man.get("background_rgb") or BACKGROUND_RGB,
                              dtype=arr.dtype)
    with Image.open(os.path.join(lib_dir, rec["file"])) as im:
        tile = np.asarray(im.convert("RGB"))
    ox, oy = rec.get("content_origin_px") or (0, 0)
    stored[oy:oy + tile.shape[0], ox:ox + tile.shape[1]] = tile

    d = np.abs(arr.astype(np.int16) - stored.astype(np.int16))
    return rec, int(d.max()), float(d.mean())


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


def servable_pose(manifest, tx=0.0, ty=0.0, tz=0.0, angle_deg=0.0, zoom=1.0):
    """The nearest pose this library can actually show, plus what was clamped.

    Returns `({tx, ty, tz, zoom}, note)`.  `note` is None when the requested
    pose was already servable.

    A clamped pose is indistinguishable from a correct one once it is on
    screen, so a UI that keeps showing the number the operator typed while the
    image sits at the cap is lying.  This reports the number that matches the
    picture.

    The clamp itself is not reimplemented here: `pose_crop` is called and its
    box inverted back into motor coordinates, so the two can never disagree.
    Depth (`w`, the component along the view axis) is not clamped -- it only
    defocuses -- so it is carried through unchanged.
    """
    zmin, zmax = zoom_limits(manifest)
    # pose_crop must see the ORIGINAL zoom or it cannot report that it clamped
    # it; a non-positive zoom would raise there, so it becomes a tiny positive
    # one that clamps to zmin and is reported like any other out-of-range zoom.
    zoom_in = float(zoom) if float(zoom) > 0.0 else 1e-9
    z = min(max(zoom_in, zmin), zmax)
    box, (W, H), _sigma, note = pose_crop(
        manifest, tx=tx, ty=ty, tz=tz, angle_deg=angle_deg, zoom=zoom_in,
        clamp=True)

    rnd = manifest["rendered"]
    win = manifest["window_mm"]
    S = float(manifest["supersample"])
    tpl_px = float(rnd["pixel_size"])
    RWp, RHp = int(rnd["width"]), int(rnd["height"])

    scale = S / z
    span_w, span_h = W * scale, H * scale
    left, upper = box[0], box[1]
    # Invert pose_crop's box construction (see its comment on the half-pixel
    # edge origin): left = RWp/2 - u/tpl_px - span_w/2 - 0.5*scale + 0.5
    u = (RWp / 2.0 - (left + span_w / 2.0 + 0.5 * scale - 0.5)) * tpl_px
    v = (RHp / 2.0 - (upper + span_h / 2.0 + 0.5 * scale - 0.5)) * tpl_px
    u -= win["centre_x"]
    v -= win["centre_y"]

    th = math.radians(angle_deg)
    w = ty * math.sin(th) + tz * math.cos(th)     # depth: never clamped
    return ({"tx": u,
             "ty": v * math.cos(th) + w * math.sin(th),
             "tz": -v * math.sin(th) + w * math.cos(th),
             "zoom": z},
            note)
