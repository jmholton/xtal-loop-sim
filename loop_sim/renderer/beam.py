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

For each mini-beam the scene path_segments() method returns the ordered list
of (material, length) segments the ray crosses, front-to-back.  The illuminated
volume contributed by one mini-beam is:

    dV_material = path_length_material × beam_weight × pixel_area

Summed over all mini-beams: illuminated volume per material (mm³, weighted).

X-ray attenuation (Beer-Lambert)
--------------------------------
Each mini-beam carries incident flux equal to its (normalised) beam weight.
Walking its segments front-to-back, the flux is attenuated in every material:

    flux_after = flux_before × exp(-mu_xray × length)

so a segment behind an absorber sees the already-attenuated flux — the metal
pin (mu_xray≈100 mm⁻¹) shadows everything downstream.  The flux absorbed in a
segment, flux_before × (1 − exp(-mu_xray × length)), is accumulated per material
as ``absorbed_dose`` (a relative dose proxy, not absolute dosimetry — mu_xray
values are illustrative).  ``transmitted_frac`` is the mean fraction of flux
that survives passing through a material; the top-level ``beam_transmission``
is the fraction of the whole incident beam that exits the sample.  By energy
book-keeping, Σ_materials absorbed_dose + beam_transmission ≈ 1.

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
          "volume_mm3"       : float,  # unweighted path-length × area sum
          "weighted_volume"  : float,  # beam-intensity-weighted sum
          "absorbed_dose"    : float,  # incident-weighted X-ray flux absorbed
                                       #   here (shadowing-aware, relative units)
          "transmitted_frac" : float,  # mean fraction of flux surviving this
                                       #   material (1.0 if never traversed)
          "fiber_axes"       : list    # only for nylon-like tube objects
        }
    plus a top-level key:
          "beam_transmission": float   # fraction of the whole incident beam
                                       #   that exits the sample
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
    # Trace all mini-beams (ordered segments → Beer-Lambert attenuation)
    # ------------------------------------------------------------------
    segment_lists = scene.path_segments(origins_s, dirs_s, t_max=200.0)

    # ------------------------------------------------------------------
    # Accumulate volumes + X-ray attenuation
    #
    # Each mini-beam carries incident flux w_i = weights_norm[i] (Σ w_i = 1).
    # Walking its segments front-to-back, exp(-mu_xray·L) attenuates the flux
    # so a segment behind an absorber sees the already-reduced flux.  Energy
    # book-keeping: Σ_materials absorbed_dose + beam_transmission ≈ 1.
    # ------------------------------------------------------------------
    def _new_entry():
        return {"volume_mm3": 0.0, "weighted_volume": 0.0,
                "absorbed_dose": 0.0, "fiber_axes": [],
                "_tfrac_wsum": 0.0, "_tfrac_w": 0.0}

    results = {}
    for obj in scene.objects:
        results.setdefault(obj.material.name, _new_entry())
    results.setdefault("air", _new_entry())

    beam_transmitted = 0.0   # Σ_i transmitted flux (already incident-weighted)

    for i, segs in enumerate(segment_lists):
        w_i  = weights_norm[i]
        flux = w_i
        per_mat = {}   # name -> [Material, total_length_in_this_beam]
        for mat, length in segs:
            name = mat.name
            if name not in per_mat:
                per_mat[name] = [mat, 0.0]
            per_mat[name][1] += length
            # Beer-Lambert over this segment, in front-to-back order
            t_seg    = np.exp(-mat.mu_xray * length)
            absorbed = flux * (1.0 - t_seg)
            if name not in results:
                results[name] = _new_entry()
            results[name]["absorbed_dose"] += absorbed
            flux *= t_seg
        beam_transmitted += flux

        # Volumes + per-material transmission (depend on per-beam totals only)
        for name, (mat, length) in per_mat.items():
            entry = results[name]
            entry["volume_mm3"]      += length * pixel_area
            entry["weighted_volume"] += length * pixel_area * w_i
            entry["_tfrac_wsum"]     += w_i * np.exp(-mat.mu_xray * length)
            entry["_tfrac_w"]        += w_i

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

    # ------------------------------------------------------------------
    # Finalise per-material entries (transmitted_frac; drop temps/empties)
    # ------------------------------------------------------------------
    for name in results:
        entry = results[name]
        w    = entry.pop("_tfrac_w")
        wsum = entry.pop("_tfrac_wsum")
        entry["transmitted_frac"] = (wsum / w) if w > 0.0 else 1.0
        if not entry["fiber_axes"]:
            entry.pop("fiber_axes")

    # Overall beam transmission (Σ w_i = 1, so this is the incident-weighted
    # mean fraction of the beam that exits the sample).
    results["beam_transmission"] = float(beam_transmitted)

    return results


def beam_volumes_json(scene, goniometer):
    """Return beam volumes as a JSON string."""
    return json.dumps(compute_beam_volumes(scene, goniometer), indent=2)


# ---------------------------------------------------------------------------
# X-ray transmission map (radiograph) — numpy reference / CPU fallback
# ---------------------------------------------------------------------------

def render_xray_numpy(scene, goniometer):
    """Per-pixel X-ray transmission map (numpy reference, CPU).

    Casts one straight ray per camera pixel along the beam axis and returns an
    (H, W) float array of transmission ``exp(-Σ mu_xray·L)`` in [0, 1]
    (1 = fully transmitted, 0 = fully absorbed).  Registered to the optical
    view (same focal-point grid as the renderer), so the metal pin / crystal
    cast a dark shadow exactly where they sit in the bright-field image.

    Order is irrelevant for transmission (the sum commutes), so this reuses
    ``path_lengths``.  It is slow at full resolution — the per-ray Python loop
    in ``path_lengths`` dominates — so the GPU ``render_xray_torch`` is the live
    path; reduce resolution for CPU snapshots.
    """
    cam = scene.camera_cfg
    W = int(cam.get("width", 640))
    H = int(cam.get("height", 480))
    pixel_size = float(cam.get("pixel_size", 0.005))
    eff_px = pixel_size / goniometer.zoom

    g = scene.geometry
    fast = np.array(g.get("camera_fast", [1, 0, 0]), dtype=float)
    slow = np.array(g.get("camera_slow", [0, 1, 0]), dtype=float)
    beam = np.array(g.get("beam_axis",   [0, 0, 1]), dtype=float)
    beam /= np.linalg.norm(beam) + 1e-30

    px = (np.arange(W) - W / 2.0) * eff_px
    py = (np.arange(H) - H / 2.0) * eff_px
    gx, gy = np.meshgrid(px, py)
    focal_pts = (gx[:, :, None] * fast + gy[:, :, None] * slow).reshape(-1, 3)
    focal_pts -= 50.0 * beam      # start 50 mm upstream

    T_inv     = goniometer.transform_inv()
    origins_s = apply_transform(T_inv, focal_pts)
    dirs_lab  = np.broadcast_to(beam, focal_pts.shape).copy()
    dirs_s    = apply_transform_dirs(T_inv, dirs_lab)

    pls = scene.path_lengths(origins_s, dirs_s, t_max=200.0)
    tau = np.array([sum(mat.mu_xray * length for mat, length in d.items())
                    for d in pls])
    return np.exp(-tau).reshape(H, W)
