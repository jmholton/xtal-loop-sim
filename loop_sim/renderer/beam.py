"""
X-ray beam volume reporter.

Beam model
----------
The beam is described by a PNG image (one pixel = one mini-beam sampling
point; pixel value = relative intensity).  The physical pixel size is stored
in the PNG as a ``pixel_size_mm`` text chunk written by make_beam_image.py.

A circular pinhole aperture (``pinhole_diameter`` in the beam config) is
applied to the loaded image at simulation time so the aperture can be changed
without regenerating the image.

For each mini-beam the scene path_lengths() method returns how far the ray
travels through each material.  The illuminated volume from one mini-beam is:

    dV_material = path_length_material × beam_weight × pixel_area

Summed over all mini-beams: illuminated volume per material (mm³, weighted).

Nylon fiber axes
----------------
For tube objects with is_fiber=True the reporter also accumulates the local
fiber axis direction (tangent to the Neville curve) weighted by the
illuminated volume of that segment, for downstream diffraction orientation
calculations.

beam: section of template.yaml
-------------------------------
    beam:
      image:            beam.png        # path to beam profile PNG
      pinhole_diameter: 0.100           # mm — circular aperture applied at load time
      # pixel_size_mm: 0.001           # override if not embedded in the PNG
"""
import json
import numpy as np

from ..motors.goniometer import apply_transform, apply_transform_dirs
from ..scene.tube        import Tube

_INF = np.inf


# ---------------------------------------------------------------------------
# Image loader
# ---------------------------------------------------------------------------

def _load_beam_image(path, pinhole_mm=0.0, fallback_px_mm=0.001):
    """
    Load a beam profile PNG and return the mini-beam grid.

    Each nonzero pixel becomes one mini-beam sampling point.

    Parameters
    ----------
    path           : str   — path to the PNG (16-bit or 8-bit greyscale/RGB)
    pinhole_mm     : float — circular aperture diameter in mm (0 = no mask)
    fallback_px_mm : float — pixel size to use if not embedded in the PNG

    Returns
    -------
    gx      : (N,) mm — horizontal offsets from beam centre (camera_fast dir)
    gy      : (N,) mm — vertical offsets from beam centre (camera_slow dir)
    weights : (N,) — relative intensities, normalised to [0, 1]
    px_mm   : float — physical pixel size in mm
    """
    from PIL import Image

    pil   = Image.open(path)
    px_mm = float(pil.info.get("pixel_size_mm", fallback_px_mm))

    arr = np.array(pil).astype(float)
    if arr.ndim > 2:
        arr = arr.mean(axis=-1)          # colour → luminance
    mx = arr.max()
    if mx > 0:
        arr /= mx                        # normalise to [0, 1]

    H, W   = arr.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    iy_g, ix_g = np.mgrid[0:H, 0:W]
    gx_all = ((ix_g - cx) * px_mm).ravel()   # horizontal (camera_fast)
    gy_all = ((iy_g - cy) * px_mm).ravel()   # vertical   (camera_slow)
    w_all  = arr.ravel()

    # Apply circular pinhole mask (zero out pixels outside the aperture)
    if pinhole_mm > 0:
        r2_mask = gx_all**2 + gy_all**2 > (pinhole_mm / 2.0)**2
        w_all[r2_mask] = 0.0

    mask = w_all > 0.0
    return gx_all[mask], gy_all[mask], w_all[mask], px_mm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
          "volume_mm3"      : float,  # unweighted path-length × area sum
          "weighted_volume" : float,  # beam-intensity-weighted sum
          "fiber_axes"      : list    # only for nylon-like tube objects
        }
    """
    bcfg = scene.beam_cfg

    g         = scene.geometry
    beam_axis = np.array(g.get("beam_axis",   [0, 0, 1]), dtype=float)
    beam_axis /= np.linalg.norm(beam_axis)
    cam_fast  = np.array(g.get("camera_fast", [1, 0, 0]), dtype=float)
    cam_slow  = np.array(g.get("camera_slow", [0, 1, 0]), dtype=float)

    # ------------------------------------------------------------------
    # Build mini-beam grid (gx, gy in mm; weights; pixel area in mm²)
    # ------------------------------------------------------------------
    img_path = bcfg.get("image", None)

    if img_path is not None:
        pinhole_mm  = float(bcfg.get("pinhole_diameter", 0.0))
        fallback_px = float(bcfg.get("pixel_size_mm", 0.001))
        gx_flat, gy_flat, weights, px_mm = _load_beam_image(
            img_path, pinhole_mm=pinhole_mm, fallback_px_mm=fallback_px)
        pixel_area   = px_mm ** 2
        # Extent used for the fiber-axes in-beam test
        extent_x = 2.0 * (np.abs(gx_flat).max() if len(gx_flat) else 0.0)
        extent_y = 2.0 * (np.abs(gy_flat).max() if len(gy_flat) else 0.0)

    else:
        # ---- Legacy inline-parameter mode ----
        spacing  = float(bcfg.get("spacing",  0.001))
        profile  = bcfg.get("profile", "flat")
        fwhm_x   = float(bcfg.get("fwhm_x", 0.05))
        fwhm_y   = float(bcfg.get("fwhm_y", 0.03))
        extent_x = 3.0 * fwhm_x
        extent_y = 3.0 * fwhm_y

        xs = np.arange(-extent_x / 2, extent_x / 2 + spacing / 2, spacing)
        ys = np.arange(-extent_y / 2, extent_y / 2 + spacing / 2, spacing)
        gx_g, gy_g = np.meshgrid(xs, ys)
        gx_flat = gx_g.ravel()
        gy_flat = gy_g.ravel()

        if profile == "gaussian":
            sx = fwhm_x / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            sy = fwhm_y / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            weights = np.exp(-gx_flat**2 / (2 * sx**2)
                             - gy_flat**2 / (2 * sy**2))
        else:
            weights = np.ones(len(gx_flat))

        pixel_area = spacing ** 2

    # Normalise weights so weighted_volume is a fraction of total
    w_sum = weights.sum()
    weights_norm = weights / (w_sum + 1e-30)

    # ------------------------------------------------------------------
    # Transform mini-beam origins to sample frame
    # ------------------------------------------------------------------
    origins_lab = (gx_flat[:, None] * cam_fast[None, :]
                   + gy_flat[:, None] * cam_slow[None, :])
    origins_lab -= 50.0 * beam_axis      # start 50 mm upstream

    T_inv      = goniometer.transform_inv()
    origins_s  = apply_transform(T_inv, origins_lab)
    dirs_lab   = np.broadcast_to(beam_axis, origins_lab.shape).copy()
    dirs_s     = apply_transform_dirs(T_inv, dirs_lab)

    # ------------------------------------------------------------------
    # Trace all mini-beams
    # ------------------------------------------------------------------
    path_length_lists = scene.path_lengths(origins_s, dirs_s, t_max=200.0)

    # ------------------------------------------------------------------
    # Accumulate volumes
    # ------------------------------------------------------------------
    results = {}
    for obj in scene.objects:
        results[obj.material.name] = {"volume_mm3": 0.0,
                                      "weighted_volume": 0.0,
                                      "fiber_axes": []}
    results.setdefault("air", {"volume_mm3": 0.0, "weighted_volume": 0.0,
                                "fiber_axes": []})

    for i, pls in enumerate(path_length_lists):
        for mat, length in pls.items():
            name = mat.name
            if name not in results:
                results[name] = {"volume_mm3": 0.0, "weighted_volume": 0.0,
                                 "fiber_axes": []}
            results[name]["volume_mm3"]      += length * pixel_area
            results[name]["weighted_volume"] += length * pixel_area * weights_norm[i]

    # ------------------------------------------------------------------
    # Fiber axis collection (for nylon diffraction orientation)
    # ------------------------------------------------------------------
    T_fwd = goniometer.transform()
    for sobj, tube in scene.fiber_objects():
        mat_name  = sobj.material.name
        midpts    = tube.segment_midpoints()   # (K, 3) in sample frame
        tangents  = tube.segment_tangents()    # (K, 3)
        midpts_lab = apply_transform(T_fwd, midpts)
        fast_proj  = midpts_lab @ cam_fast
        slow_proj  = midpts_lab @ cam_slow
        in_beam    = ((np.abs(fast_proj) < extent_x / 2) &
                      (np.abs(slow_proj) < extent_y / 2))
        for k in range(len(midpts)):
            if in_beam[k]:
                seg_len = np.linalg.norm(
                    tube._curve_pts[k + 1] - tube._curve_pts[k])
                vol = np.pi * tube.radius**2 * seg_len
                results[mat_name]["fiber_axes"].append({
                    "axis":   tangents[k].tolist(),
                    "volume": vol,
                })

    # Drop empty fiber_axes lists
    for name in results:
        if not results[name]["fiber_axes"]:
            results[name].pop("fiber_axes")

    return results


def beam_volumes_json(scene, goniometer):
    """Return beam volumes as a JSON string."""
    return json.dumps(compute_beam_volumes(scene, goniometer), indent=2)
