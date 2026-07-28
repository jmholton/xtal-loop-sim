"""
Pre-computed frame library: a rotation sweep rendered once and replayed.

Rationale
---------
The camera is orthographic, so translating the sample sideways shifts the
image by an exact whole number of pixels and nothing else (verified: max
pixel difference 0.0).  Panning therefore needs no render at all -- it is a
crop out of a larger frame.  Rotation is the only motor that genuinely
changes image content, so a single sweep about the spindle covers the
interactive envelope:

    frames rendered  = 360 / step_deg          (e.g. 360 at 1 deg)
    tx / ty          = crop offset into the rendered margin, free
    zoom / tz        = NOT covered -- still needs a live render

Each frame is rendered `margin`x larger than the camera so there is material
to pan into.  `pan_px` in the manifest is how far the crop window may move
from centre before it runs off the rendered area.

Layout
------
    frame_library/<scene_stem>/manifest.json
    frame_library/<scene_stem>/rot_0000.jpg ...

The manifest records a SHA-256 of the scene YAML; `ensure_library()` rebuilds
when the scene changes, so a new or edited scene regenerates on first use.
"""
import hashlib
import io
import json
import os
import time

import numpy as np

# Anchored to the repo root rather than the caller's cwd: a library written to
# the wrong directory is a silently missing deliverable, since it has to land in git.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ROOT = os.path.join(_REPO_ROOT, "frame_library")
DEFAULT_STEP_DEG = 1.0
DEFAULT_MARGIN = 1.5
DEFAULT_N_COND = 7
DEFAULT_QUALITY = 90


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


def is_current(scene_path, lib_dir):
    """True when a complete library matching the current scene is on disk."""
    man = load_manifest(lib_dir)
    if man is None:
        return False
    if man.get("scene_sha256") != scene_fingerprint(scene_path):
        return False
    return all(os.path.exists(os.path.join(lib_dir, f["file"]))
               for f in man["frames"])


def build_library(scene_path, root=DEFAULT_ROOT, axis="rotx",
                  step_deg=DEFAULT_STEP_DEG, margin=DEFAULT_MARGIN,
                  n_cond=DEFAULT_N_COND, quality=DEFAULT_QUALITY,
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

    lib_dir = library_dir(scene_path, root)
    os.makedirs(lib_dir, exist_ok=True)

    scene = load(scene_path, device="cpu")
    cam = scene.camera_cfg
    W, H = int(cam["width"]), int(cam["height"])
    RW, RH = int(round(W * margin)), int(round(H * margin))
    cam["width"], cam["height"] = RW, RH

    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    tscene = TorchScene(scene, dev, torch.float64)

    angles = [round(i * step_deg, 6) for i in range(int(round(360.0 / step_deg)))]
    frames = []
    t_start = time.time()
    for i, ang in enumerate(angles):
        gono = Goniometer(scene.geometry)
        gono.set(**{axis: ang})
        try:
            img = render_torch(tscene, gono, n_cond=n_cond)
        except torch.OutOfMemoryError as exc:
            raise RuntimeError(
                f"Out of memory rendering {scene_path} at {RW}x{RH}, n_cond={n_cond}. "
                f"Mesh-bearing scenes (solvent droplets) need the TSurfaceMesh AABB "
                f"cull before a library can be built at this size -- see "
                f"docs/HANDOFF.md risk A. Reduce --margin or --n-cond to proceed."
            ) from exc
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
        "jpeg_quality": quality,
        "camera": {"width": W, "height": H,
                   "pixel_size": cam["pixel_size"],
                   "na_objective": cam["na_objective"],
                   "na_condenser": cam["na_condenser"]},
        "rendered": {"width": RW, "height": RH, "margin": margin},
        "pan_px": {"x": (RW - W) // 2, "y": (RH - H) // 2},
        "frames": frames,
    }
    with open(os.path.join(lib_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def ensure_library(scene_path, root=DEFAULT_ROOT, progress=print, **kwargs):
    """Return the manifest, building the library first if it is absent or stale."""
    lib_dir = library_dir(scene_path, root)
    if is_current(scene_path, lib_dir):
        return load_manifest(lib_dir)
    if progress:
        progress(f"[frame-library] no current library for {scene_path} -- building")
    return build_library(scene_path, root=root, progress=progress, **kwargs)


def frame_for_angle(manifest, angle_deg):
    """Nearest pre-rendered frame record for an arbitrary spindle angle."""
    step = manifest["step_deg"]
    idx = int(round((angle_deg % 360.0) / step)) % len(manifest["frames"])
    return manifest["frames"][idx]


def crop_window(manifest, tx_mm=0.0, ty_mm=0.0):
    """Pixel crop box (left, upper, right, lower) for a sample translation.

    Sideways translation is an exact image shift under this orthographic
    camera, so panning is a crop rather than a render.  Raises ValueError
    when the requested pan exceeds the rendered margin.
    """
    cam, rnd = manifest["camera"], manifest["rendered"]
    px = cam["pixel_size"]
    dx, dy = int(round(tx_mm / px)), int(round(ty_mm / px))
    lim = manifest["pan_px"]
    if abs(dx) > lim["x"] or abs(dy) > lim["y"]:
        raise ValueError(
            f"pan ({dx},{dy}) px exceeds rendered margin "
            f"({lim['x']},{lim['y']}) px -- rebuild with a larger --margin")
    left = (rnd["width"] - cam["width"]) // 2 + dx
    upper = (rnd["height"] - cam["height"]) // 2 + dy
    return (left, upper, left + cam["width"], upper + cam["height"])
