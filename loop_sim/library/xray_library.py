"""
Pre-computed X-ray radiograph library: a rotation sweep rendered once and
replayed. The X-ray analogue of frame_library.py.

Why this is a SEPARATE module, not a parameter on frame_library.py: see
docs/DECISIONS.md 2026-08-18 and the `_RENDER_SOURCES` comment in
frame_library.py -- `render_xray_torch`/`trace_xray` used to live inside
`engine_torch.py` (which `frame_library.py` DOES hash), so an X-ray-only GPU
edit was silently invalidating every optical library. Splitting the render
code out (`renderer/xray_torch.py`) and the library-build code out (this
file) means the two hash scopes -- and the two library trees -- can never
leak into each other again.

Simpler than the optical case in three ways that shaped this file:
  * The radiograph needs no `n_cond` condenser loop and no PSF -- the beam is
    collimated, not a Kohler illumination cone, so there is nothing to
    soft-edge.
  * Depth (`tz`) genuinely does not change a collimated Beer-Lambert integral
    (translating a ray's start point along its own direction cannot change
    which materials an infinite line crosses), so unlike `pose_crop`'s
    PSF-derived defocus blur there is nothing to approximate -- storing
    `na_condenser: 0.0` in the manifest's `camera` block makes `pose_crop`
    (reused verbatim) compute `sigma_px == 0.0` for every pose, for free,
    with no reader-side special case.
  * The content is a continuous-tone transmission map, not a near-binary
    photograph, so it is stored as 16-bit greyscale rather than 8-bit RGB --
    the pin transmits at ~1e-31 while the biological signal of interest sits
    in roughly the top fifth of the range, and 8-bit would quantize that
    away. Because `AIR.mu_xray == 0.0` exactly, background transmission is
    exactly 1.0 -> exactly 65535 at 16-bit, so `frame_library.py`'s
    EXACT-equality `content_bbox`/`crop_to_content` transfer unmodified:
    this module reshapes the (H, W) transmission array to (H, W, 1) before
    calling them (both already handle any channel count via
    `arr.min(axis=2)`), then squeezes it back for the actual PNG save.

Reused from frame_library.py, unmodified: `crop_margin_px`, `content_bbox`,
`crop_to_content`, `_round_up_to_parity`, `plan_window`, `scene_fingerprint`,
`_sha_over`, `load_manifest`, `_write_manifest`, `library_dir` (with a
different root), `cuda_available`, `CPU_BUILD_REFUSAL` -- and, because they
operate purely on manifest dict fields and never on pixel format,
`zoom_limits`, `frame_for_angle`, `pose_crop`, `servable_pose`.

Not reused, reimplemented here with the same shape: `build_params` (no
n_cond/psf/jpeg_quality to resolve), `library_diff`/`library_status`/
`is_current` (loop over this module's own build keys, not the optical ones),
`content_window` (scouts `render_xray_torch` instead of `render_torch`, and
masks a scalar map instead of RGB), `build_library` (calls
`render_xray_torch`, saves 16-bit greyscale PNG instead of 8-bit RGB).
"""
import glob
import math
import os
import time

import numpy as np

from .frame_library import (
    CPU_BUILD_REFUSAL, DEFAULT_PAN_MM, DEFAULT_STEP_DEG, DEFAULT_SUPERSAMPLE,
    MAX_TEMPLATE_MPX, PNG_COMPRESS_LEVEL, _round_up_to_parity, _sha_over,
    _write_manifest, content_bbox, crop_margin_px, crop_to_content,
    cuda_available, frame_for_angle, library_dir, load_manifest, plan_window,
    pose_crop, scene_fingerprint, servable_pose, zoom_limits,
)

__all__ = [
    "DEFAULT_ROOT", "BACKGROUND_I16", "xray_render_sha",
    "xray_render_source_paths", "xray_build_params", "xray_library_diff",
    "xray_library_status", "xray_is_current", "content_window_xray",
    "build_xray_library", "ensure_xray_library",
    # re-exported for callers that only want to import from this module
    "library_dir", "load_manifest", "zoom_limits", "frame_for_angle",
    "pose_crop", "servable_pose", "cuda_available", "CPU_BUILD_REFUSAL",
]

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ROOT = os.path.join(_REPO_ROOT, "xray_library")

# Background (AIR.mu_xray == 0.0) is EXACTLY 1.0 transmission -> exactly
# 65535 at 16-bit. A 1-tuple so frame_library.content_bbox/crop_to_content's
# "uniform background" fast path (`arr.min(axis=2) < bg[0]`) applies
# unmodified against a (H, W, 1)-reshaped transmission map.
BACKGROUND_I16 = (65535,)

# Deliberately NOT disjoint from frame_library._RENDER_SOURCES on
# engine_torch.py: trace_xray calls tscene.next_interface(), which lives
# there, so a correctness fix to interface detection changes X-ray output
# too and must invalidate this library, same as the optical one (this is
# exactly the coupling docs/DECISIONS.md 2026-08-18 calls out -- the landmine
# the split fixed was X-RAY-only edits invalidating the OPTICAL library, not
# this direction, which is correct and intentional). beam.py is the numpy
# reference -- not called by this GPU build path, but it IS the fallback the
# server serves from with no GPU, so a change to it should still mark an
# X-ray library's provenance stale, consistent with how the optical side
# treats its numpy/torch reference pair as one unit.
_XRAY_RENDER_SOURCES = ("renderer/xray_torch.py", "renderer/engine_torch.py",
                        "renderer/beam.py", "scene/*.py", "motors/goniometer.py")
_xray_render_sha_cache = None


def xray_render_source_paths():
    """`(package_root, sorted_paths)` -- exactly the files `xray_render_sha`
    hashes. Public for the same reason `frame_library.render_source_paths`
    is: a test should be able to assert which files are covered.
    """
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = []
    for pat in _XRAY_RENDER_SOURCES:
        paths.extend(glob.glob(os.path.join(pkg, *pat.split("/"))))
    return pkg, sorted(paths)


def xray_render_sha():
    """sha256 over the source of every module that decides an X-ray template
    pixel. Cached per process, same rationale as frame_library.render_sha.
    """
    global _xray_render_sha_cache
    if _xray_render_sha_cache is None:
        _xray_render_sha_cache = _sha_over(*xray_render_source_paths())
    return _xray_render_sha_cache


# Build parameters that change the pixels. No n_cond/psf/jpeg_quality/format:
# the tracer has no condenser loop and no PSF, and storage is always
# lossless 16-bit PNG -- there is no lossy alternative worth offering.
_XRAY_BUILD_KEYS = ("axis", "step_deg", "supersample", "pan_mm", "render_sha")


def xray_build_params(axis="rotx", step_deg=DEFAULT_STEP_DEG,
                      supersample=DEFAULT_SUPERSAMPLE, pan_mm=DEFAULT_PAN_MM,
                      **_ignored):
    """The pixel-affecting settings, keyed as the manifest stores them."""
    return {"axis": axis, "step_deg": step_deg, "supersample": supersample,
            "pan_mm": pan_mm, "render_sha": xray_render_sha()}


def xray_library_diff(scene_path, lib_dir, **params):
    """Which BUILD PARAMETERS a library on disk disagrees with `params`
    about. Mirrors frame_library.library_diff exactly, over
    `_XRAY_BUILD_KEYS` instead of the optical `_BUILD_KEYS`.
    """
    man = load_manifest(lib_dir)
    if man is None or not man.get("frames"):
        return {}
    out = {}
    for key in _XRAY_BUILD_KEYS:
        want = params.get(key)
        if want is None:
            continue
        have = man.get(key)
        if have != want:
            out[key] = {"have": have, "want": want}
    return out


def _xray_frames_complete(lib_dir, man):
    """True when every frame the manifest lists is on disk and the right
    size. Mirrors frame_library._frames_complete's size-vs-manifest guard
    (PIL's Image.open works identically for 16-bit greyscale); kept as its
    own copy rather than importing a private helper, since the two trees'
    frame formats are allowed to diverge further later.
    """
    if not man.get("frames"):
        return False
    if not all(os.path.exists(os.path.join(lib_dir, f["file"]))
               for f in man["frames"]):
        return False
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


def xray_library_status(scene_path, lib_dir, **params):
    """'current' | 'stale' | 'missing' for an X-ray library on disk. Mirrors
    frame_library.library_status exactly.
    """
    man = load_manifest(lib_dir)
    if man is None:
        return "missing"
    if man.get("scene_sha256") != scene_fingerprint(scene_path):
        return "missing"
    if not _xray_frames_complete(lib_dir, man):
        return "missing"
    return "stale" if xray_library_diff(scene_path, lib_dir, **params) else "current"


def xray_is_current(scene_path, lib_dir, **params):
    return xray_library_status(scene_path, lib_dir, **params) == "current"


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------
def content_window_xray(scene, tscene, axis="rotx", n_probe=8, fov_mult=4.0,
                        scout_div=2, eps=1e-3, max_fov_mult=64.0,
                        progress=None):
    """Image-plane bounding box of visible content, in mm, over a full sweep.

    `frame_library.content_window`'s X-ray twin: same scout-and-widen
    algorithm, but scouts `render_xray_torch` (a single (H, W) transmission
    map, no RGB) and a tighter `eps` -- there is no PSF to smear a hard edge
    across several levels, so the optical default (2/255 = 0.0078) is too
    coarse for a scalar map whose useful contrast is a few percent.
    """
    from ..motors.goniometer import Goniometer
    from ..renderer.xray_torch import render_xray_torch

    cam = scene.camera_cfg
    W0, H0 = int(cam["width"]), int(cam["height"])
    px0 = float(cam["pixel_size"])
    saved = (cam["width"], cam["height"], cam["pixel_size"])

    try:
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
                img = render_xray_torch(tscene, gono).cpu().numpy()   # (H, W)
                border = np.concatenate([img[0], img[-1]])
                bg = float(np.median(border))
                mask = np.abs(img - bg) > eps
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
                    progress(f"[xray-library] WARNING content still reaches the "
                             f"scout edge at {fov_mult:g}x the field of view -- "
                             f"the render window may be truncated")
                break
            fov_mult *= 2.0
            if progress:
                progress(f"[xray-library] content reaches the scout edge; "
                         f"widening the survey to {fov_mult:g}x the field of view")
    finally:
        cam["width"], cam["height"], cam["pixel_size"] = saved

    if not math.isfinite(x0):          # nothing visible; fall back to the camera
        return (-W0 * px0 / 2, W0 * px0 / 2, -H0 * px0 / 2, H0 * px0 / 2)
    return (x0, x1, y0, y1)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_xray_library(scene_path, root=DEFAULT_ROOT, axis="rotx",
                       step_deg=DEFAULT_STEP_DEG, supersample=DEFAULT_SUPERSAMPLE,
                       pan_mm=DEFAULT_PAN_MM, tile_size=250_000, device=None,
                       progress=print):
    """Render a full 360deg X-ray radiograph sweep about `axis` and write it
    to disk.

    Mirrors `frame_library.build_library`, simplified throughout because the
    tracer has no `n_cond` loop, no PSF, and no VRAM-preflight need -- the
    default `tile_size` is already conservative and every scene in this repo,
    including the heaviest mesh, renders comfortably under it (measured:
    ~0.87s/frame on `hampton_300um_realistic`, 640x480, RTX 4080S).
    """
    import torch
    from PIL import Image
    from ..scene.scene import load
    from ..motors.goniometer import Goniometer
    from ..renderer.torch_compat import ensure_dynamo
    ensure_dynamo()   # torch 2.0.1 does not bind torch._dynamo itself
    from ..renderer.engine_torch import TorchScene
    from ..renderer.xray_torch import render_xray_torch

    if axis != "rotx":
        # build_xray_library's window offset and pose_crop both hard-code the
        # rotx coupling, same as the optical build -- see its identical guard.
        raise ValueError(
            f"axis={axis!r} is not supported yet -- the window offset and "
            f"pose_crop both assume the spindle is rotx. Generalise both "
            f"before enabling another axis.")

    lib_dir = library_dir(scene_path, root)
    os.makedirs(lib_dir, exist_ok=True)
    man_path = os.path.join(lib_dir, "manifest.json")
    if os.path.exists(man_path):
        os.remove(man_path)
    for old in sorted(os.listdir(lib_dir)):
        if old.startswith("rot_") and old.lower().endswith(".png"):
            os.remove(os.path.join(lib_dir, old))

    scene = load(scene_path, device="cpu")
    cam = scene.camera_cfg
    W, H = int(cam["width"]), int(cam["height"])
    px0 = float(cam["pixel_size"])

    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    tscene = TorchScene(scene, dev, torch.float64)
    if progress:
        progress(f"[xray-library] device {dev}")

    # --- framing: measure the scene, then centre the window on it -----------
    content = content_window_xray(scene, tscene, axis=axis, progress=progress)
    wx0, wx1, wy0, wy1 = plan_window(cam, content, pan_mm=pan_mm)
    tpl_px = px0 / supersample
    RW = _round_up_to_parity(math.ceil((wx1 - wx0) / tpl_px), W)
    RH = _round_up_to_parity(math.ceil((wy1 - wy0) / tpl_px), H)
    if RW * RH > MAX_TEMPLATE_MPX * 1e6:
        raise RuntimeError(
            f"refusing to build a {RW}x{RH} ({RW*RH/1e6:.0f} Mpx) template for "
            f"{scene_path} -- over the {MAX_TEMPLATE_MPX:g} Mpx guard. Lower "
            f"--supersample or --pan-mm, or raise MAX_TEMPLATE_MPX.")
    wx1, wy1 = wx0 + RW * tpl_px, wy0 + RH * tpl_px
    x_win = 0.5 * (wx0 + wx1)
    y_win = 0.5 * (wy0 + wy1)
    if progress:
        progress(f"[xray-library] content x[{content[0]:.3f},{content[1]:.3f}] "
                 f"y[{content[2]:.3f},{content[3]:.3f}] mm")
        progress(f"[xray-library] window {wx1-wx0:.3f} x {wy1-wy0:.3f} mm "
                 f"centred ({x_win:.3f},{y_win:.3f}) -> {RW}x{RH} px "
                 f"({RW*RH/1e6:.2f} Mpx, supersample {supersample}x)")

    cam["width"], cam["height"], cam["pixel_size"] = RW, RH, tpl_px

    # na_condenser: 0.0 -- read by pose_crop (reused verbatim) to compute the
    # defocus blur; zero because the beam is collimated and tz genuinely does
    # not defocus a radiograph. See the module docstring.
    geom = {"camera": {"width": W, "height": H, "pixel_size": px0,
                       "na_condenser": 0.0},
            "rendered": {"width": RW, "height": RH, "pixel_size": tpl_px},
            "window_mm": {"centre_x": x_win, "centre_y": y_win},
            "supersample": supersample}
    margin = crop_margin_px(supersample, zoom_limits(geom)[0])

    angles = [round(i * step_deg, 6) for i in range(int(round(360.0 / step_deg)))]
    prog_every = max(1, len(angles) // 20)
    frames = []
    t_start = time.time()
    for i, ang in enumerate(angles):
        gono = Goniometer(scene.geometry)
        th = math.radians(ang)
        gono.set(**{axis: ang, "tx": -x_win,
                    "ty": -y_win * math.cos(th), "tz": y_win * math.sin(th)})
        T = render_xray_torch(tscene, gono, tile_size=tile_size)
        arr16 = (T.clamp(0, 1) * 65535.0).round().cpu().numpy().astype(np.uint16)
        arr16 = arr16[..., None]                          # (H, W) -> (H, W, 1)
        arr16, (ox, oy) = crop_to_content(arr16, margin, BACKGROUND_I16)
        arr16 = arr16[..., 0]              # back to 2D -- PIL infers I;16 from it
        name = f"rot_{i:04d}.png"
        Image.fromarray(arr16).save(
            os.path.join(lib_dir, name), format="PNG",
            compress_level=PNG_COMPRESS_LEVEL)
        frames.append({"index": i, "angle_deg": ang, "file": name,
                       "content_origin_px": [ox, oy],
                       "content_size_px": [arr16.shape[1], arr16.shape[0]]})
        if progress and (i % prog_every == 0 or i == len(angles) - 1):
            el = time.time() - t_start
            progress(f"    {i+1}/{len(angles)} frames  {el:6.1f}s elapsed "
                     f"({el/(i+1):.2f}s/frame)")

    manifest = {
        "scene": scene_path,
        "scene_sha256": scene_fingerprint(scene_path),
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modality": "xray",
        "axis": axis,
        "step_deg": step_deg,
        "supersample": supersample,
        "pan_mm": pan_mm,
        "render_sha": xray_render_sha(),
        # THE VIRTUAL WINDOW, not the stored image size -- same convention as
        # the optical manifest; pose_crop/zoom_limits/servable_pose all key
        # off this. na_condenser: 0.0, see above.
        "camera": {"width": W, "height": H, "pixel_size": px0,
                   "na_condenser": 0.0},
        "rendered": {"width": RW, "height": RH, "pixel_size": tpl_px},
        "window_mm": {"x0": wx0, "x1": wx1, "y0": wy0, "y1": wy1,
                      "centre_x": x_win, "centre_y": y_win},
        "background_i16": list(BACKGROUND_I16),
        "crop_margin_px": margin,
        "frames": frames,
    }
    _write_manifest(lib_dir, manifest)
    return manifest


def ensure_xray_library(scene_path, root=DEFAULT_ROOT, progress=print, **kwargs):
    """Return the manifest, building the library first if it is absent or stale."""
    lib_dir = library_dir(scene_path, root)
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    params = xray_build_params(**kwargs)
    if xray_is_current(scene_path, lib_dir, **params):
        return load_manifest(lib_dir)
    if progress:
        progress(f"[xray-library] no current library for {scene_path} -- building")
    return build_xray_library(scene_path, root=root, progress=progress, **kwargs)
