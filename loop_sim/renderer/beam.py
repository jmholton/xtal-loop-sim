"""
X-ray beam volume reporter.

Models the X-ray beam as a regular grid of parallel mini-beams (each ~1 µm²
in cross-section), optionally weighted by a YAG scintillator image of the
actual beam profile.

For each mini-beam ray the scene path_lengths() method returns how far the
ray travels through each material.  The illuminated volume contribution from
one mini-beam is:

    dV_material = path_length_material × beam_weight × pixel_area

Summed over all mini-beams: illuminated volume per material (mm³, weighted).

Nylon fiber axes
----------------
For tube objects with is_fiber=True, the beam reporter also accumulates the
local fiber axis direction (tangent to the Neville curve) weighted by the
illuminated volume of that segment.  This allows downstream code to compute
expected nylon fiber diffraction orientations.
"""
import json
import numpy as np

try:
    import tifffile
    _HAS_TIFFFILE = True
except ImportError:
    _HAS_TIFFFILE = False

from ..motors.goniometer import apply_transform, apply_transform_dirs
from ..scene.tube        import Tube

_INF = np.inf


def _load_yag_image(path):
    """Load a YAG beam profile image and return a normalised float array."""
    if _HAS_TIFFFILE:
        img = tifffile.imread(path).astype(float)
    else:
        from PIL import Image
        img = np.array(Image.open(path)).astype(float)
    img -= img.min()
    mx = img.max()
    if mx > 0:
        img /= mx
    return img


def _gaussian_weights(grid_x, grid_y, fwhm_x, fwhm_y):
    """2-D Gaussian intensity profile on a grid."""
    sx = fwhm_x / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    sy = fwhm_y / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return np.exp(-grid_x**2 / (2 * sx**2) - grid_y**2 / (2 * sy**2))


def compute_beam_volumes(scene, goniometer):
    """
    Compute illuminated volumes per material.

    Parameters
    ----------
    scene      : Scene (from scene.scene.load)
    goniometer : Goniometer (current motor positions)

    Returns
    -------
    dict with keys = material names, values = sub-dicts:
        {
          "volume_mm3"     : float,   # unweighted path-length × area sum
          "weighted_volume" : float,  # beam-intensity-weighted sum
          "fiber_axes"     : list     # (only for nylon-like tube objects)
        }

    Also returns a "total_beam_volume_mm3" key.
    """
    bcfg = scene.beam_cfg
    spacing    = float(bcfg.get("spacing",  0.001))    # mm
    profile    = bcfg.get("profile", "flat")
    fwhm_x     = float(bcfg.get("fwhm_x", 0.05))
    fwhm_y     = float(bcfg.get("fwhm_y", 0.03))
    yag_path   = bcfg.get("profile_image", None)
    pixel_area = spacing ** 2                           # mm²

    g = scene.geometry
    beam_axis  = np.array(g.get("beam_axis",    [0, 0, 1]), dtype=float)
    beam_axis /= np.linalg.norm(beam_axis)
    cam_fast   = np.array(g.get("camera_fast",  [1, 0, 0]), dtype=float)
    cam_slow   = np.array(g.get("camera_slow",  [0, 1, 0]), dtype=float)

    # --- Build grid of mini-beam origins (in lab frame, at z=0) ---
    # Grid extent: 3× the larger FWHM, or YAG image size
    extent_x = 3.0 * fwhm_x
    extent_y = 3.0 * fwhm_y

    xs = np.arange(-extent_x / 2, extent_x / 2 + spacing / 2, spacing)
    ys = np.arange(-extent_y / 2, extent_y / 2 + spacing / 2, spacing)
    gx, gy = np.meshgrid(xs, ys)
    gx_flat = gx.ravel()
    gy_flat = gy.ravel()

    # Mini-beam origins in lab frame (at z=0 plane)
    origins_lab = (gx_flat[:, None] * cam_fast[None, :]
                   + gy_flat[:, None] * cam_slow[None, :])
    # Shift upstream along beam axis so rays start before the sample
    origins_lab -= 50.0 * beam_axis

    # --- Beam intensity weights ---
    if profile == "flat":
        weights = np.ones(len(gx_flat))
    elif profile == "gaussian":
        weights = _gaussian_weights(gx_flat, gy_flat, fwhm_x, fwhm_y).ravel()
    elif profile == "image" and yag_path is not None:
        yag = _load_yag_image(yag_path)
        # Resample YAG image onto our grid
        # Assume YAG pixel size is stored in beam config
        yag_px_size = float(bcfg.get("yag_pixel_size_mm", spacing))
        Hy, Wx = yag.shape
        xi = (gx_flat / yag_px_size + Wx / 2).astype(int).clip(0, Wx - 1)
        yi = (gy_flat / yag_px_size + Hy / 2).astype(int).clip(0, Hy - 1)
        weights = yag[yi, xi]
    else:
        weights = np.ones(len(gx_flat))

    # Normalise weights so they sum to 1 (for fractional output)
    w_sum = weights.sum()
    weights_norm = weights / (w_sum + 1e-30)

    # --- Transform mini-beam rays to sample frame ---
    T_inv = goniometer.transform_inv()
    origins_s = apply_transform(T_inv, origins_lab)
    dirs_lab   = np.broadcast_to(beam_axis, origins_lab.shape).copy()
    dirs_s     = apply_transform_dirs(T_inv, dirs_lab)

    # --- Trace all mini-beams ---
    path_length_lists = scene.path_lengths(origins_s, dirs_s, t_max=200.0)

    # --- Accumulate volumes ---
    results = {}
    for mat_name in [obj.material.name for obj in scene.objects] + ["air"]:
        results[mat_name] = {"volume_mm3": 0.0, "weighted_volume": 0.0,
                             "fiber_axes": []}

    for i, pls in enumerate(path_length_lists):
        for mat, length in pls.items():
            name = mat.name
            if name not in results:
                results[name] = {"volume_mm3": 0.0, "weighted_volume": 0.0,
                                 "fiber_axes": []}
            results[name]["volume_mm3"]      += length * pixel_area
            results[name]["weighted_volume"] += length * pixel_area * weights_norm[i]

    # --- Fiber axis collection ---
    for sobj, tube in scene.fiber_objects():
        mat_name = sobj.material.name
        midpts  = tube.segment_midpoints()   # (K, 3)  — in sample frame
        tangents = tube.segment_tangents()   # (K, 3)
        # Transform midpoints to lab frame to check if in beam
        T_fwd = goniometer.transform()
        midpts_lab = apply_transform(T_fwd, midpts)
        # A segment midpoint is "in the beam" if its transverse offset < 2×extent
        fast_proj = midpts_lab @ cam_fast
        slow_proj = midpts_lab @ cam_slow
        in_beam = ((np.abs(fast_proj) < extent_x / 2) &
                   (np.abs(slow_proj) < extent_y / 2))
        for k in range(len(midpts)):
            if in_beam[k]:
                # Weight by approximate illuminated volume of segment
                seg_len = np.linalg.norm(tube._curve_pts[k + 1] - tube._curve_pts[k])
                r = tube.radius
                vol = np.pi * r**2 * seg_len
                results[mat_name]["fiber_axes"].append({
                    "axis":   tangents[k].tolist(),
                    "volume": vol,
                })

    # Remove empty fiber_axes for non-fiber materials
    for name in results:
        if not results[name]["fiber_axes"]:
            results[name].pop("fiber_axes")

    return results


def beam_volumes_json(scene, goniometer):
    """Return beam volumes as a JSON string."""
    data = compute_beam_volumes(scene, goniometer)
    return json.dumps(data, indent=2)
