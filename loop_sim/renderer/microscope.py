"""
Bright-field optical microscope renderer.

Physics model
-------------
Snell's law refraction at every material interface: a ray entering a denser
medium is bent toward the normal; if bent past the objective's collection
angle it is not collected (dark pixel).

Condenser illumination (soft NA edges)
    Each pixel is sampled by `n_cond` rays at different illumination angles,
    uniformly covering the condenser aperture disk.  The pixel brightness is
    the average collection over all illumination angles.  This naturally
    produces smooth edge contrast instead of a binary step.

    n_cond = 1  → single on-axis ray, hard NA cutoff (fast, good for preview)
    n_cond = 7  → hex grid (1 centre + 6 ring), smooth edges, 7× cost
    n_cond = 19 → two hex rings, very smooth

Beer-Lambert absorption
    Along each ray segment through a material with mu_optical > 0:
    intensity *= exp(-mu * segment_length)

Outputs
-------
render() returns a (H, W) float32 NumPy array in [0, 1] and JPEG bytes.
"""
import io
import numpy as np
from PIL import Image

from ..motors.goniometer import apply_transform, apply_transform_dirs
from ..scene.materials   import AIR

_INF = np.inf
MAX_DEPTH = 12    # max number of refractions to follow per ray


# ---------------------------------------------------------------------------
# Condenser sampling grid
# ---------------------------------------------------------------------------

def _condenser_offsets(n_cond, na_cond, n_medium=1.0):
    """
    Return (n_cond, 3) direction offsets for condenser illumination sampling.
    Directions are unit vectors; the base direction is assumed to be (0,0,1)
    and the caller rotates them to the actual optical axis.
    """
    if n_cond == 1:
        return np.array([[0.0, 0.0, 0.0]])   # on-axis only (offset = 0)

    max_angle = np.arcsin(na_cond / n_medium)

    # Hexagonal grid in angular space
    offsets = [[0.0, 0.0]]
    ring = 1
    while len(offsets) < n_cond:
        r = ring * max_angle / (np.ceil(np.sqrt(n_cond) / 2) + 1)
        k_pts = 6 * ring
        for k in range(k_pts):
            angle = 2 * np.pi * k / k_pts
            offsets.append([r * np.cos(angle), r * np.sin(angle)])
            if len(offsets) >= n_cond:
                break
        ring += 1

    offsets = np.array(offsets[:n_cond], dtype=float)   # (n_cond, 2) angular offsets
    # Convert angular offsets to direction unit vectors (small angle: sin ≈ angle)
    ox = np.sin(offsets[:, 0])
    oy = np.sin(offsets[:, 1])
    oz = np.sqrt(np.maximum(1.0 - ox**2 - oy**2, 0.0))
    return np.stack([ox, oy, oz], axis=1)               # (n_cond, 3)


# ---------------------------------------------------------------------------
# Snell's law helpers
# ---------------------------------------------------------------------------

def _snell_refract(d, n_hat, n1, n2):
    """
    Compute refracted direction vectors (vectorized).

    d     : (N, 3) incident unit directions
    n_hat : (N, 3) surface normals pointing into incident medium (outward)
    n1    : (N,)   refractive index of incident medium
    n2    : (N,)   refractive index of transmitted medium

    Returns (N, 3) refracted directions; NaN rows indicate TIR.
    """
    # Ensure n_hat points toward incident ray (same hemisphere as -d)
    flip = (np.einsum("ij,ij->i", d, n_hat) > 0)
    n_hat = n_hat.copy()
    n_hat[flip] *= -1

    cos_i = -np.einsum("ij,ij->i", d, n_hat).clip(-1, 1)  # (N,)
    r = n1 / n2
    sin2_t = r**2 * (1.0 - cos_i**2)
    tir = sin2_t > 1.0   # total internal reflection

    cos_t = np.sqrt(np.maximum(1.0 - sin2_t, 0.0))
    d_refract = (r[:, None] * d
                 + (r * cos_i - cos_t)[:, None] * n_hat)

    # Normalise
    nlen = np.linalg.norm(d_refract, axis=1, keepdims=True)
    d_refract /= nlen + 1e-30

    d_refract[tir] = np.nan
    return d_refract


def _fresnel_T(n1, n2, cos_i):
    """Unpolarised Fresnel transmittance (1 - reflectance)."""
    sin2_t = (n1 / n2)**2 * (1.0 - cos_i**2)
    tir = sin2_t >= 1.0
    cos_t = np.sqrt(np.maximum(1.0 - sin2_t, 0.0))

    rs = ((n1 * cos_i - n2 * cos_t) / (n1 * cos_i + n2 * cos_t + 1e-30))**2
    rp = ((n2 * cos_i - n1 * cos_t) / (n2 * cos_i + n1 * cos_t + 1e-30))**2
    T = 1.0 - (rs + rp) / 2.0
    T[tir] = 0.0
    return T


# ---------------------------------------------------------------------------
# Single-pass ray trace (one illumination direction per pixel)
# ---------------------------------------------------------------------------

def _trace_rays(scene, origins, dirs, na_obj, n_background=1.0,
                opt_axis_sample=None):
    """
    Trace N rays through the scene.

    Maintains a compressed array of *active* rays that shrinks each depth
    iteration as rays exit the scene or undergo TIR.  This avoids re-testing
    background rays (which exit immediately) on every subsequent bounce.

    opt_axis_sample : (3,) array — optical axis in the sample frame.
        If None, falls back to scene.geometry['optical_axis'] (correct only
        when there is no sample rotation).  Pass this explicitly whenever the
        goniometer has non-zero rotation so the NA cutoff is evaluated in the
        correct frame (camera fixed in lab; sample rotates under it).

    Returns (N,) float array: intensity in [0, 1].
    """
    N = len(origins)
    intensity = np.ones(N)

    if opt_axis_sample is not None:
        opt_axis = np.asarray(opt_axis_sample, dtype=float)
    else:
        opt_axis = np.array(scene.geometry.get("optical_axis", [0, 0, -1]),
                            dtype=float)
    cos_na = np.cos(np.arcsin(np.clip(na_obj, 0.0, 1.0)))

    # Compressed active-ray state (shrinks each iteration)
    gidx     = np.arange(N, dtype=np.intp)   # global indices of active rays
    o        = origins.copy()
    d        = dirs.copy()
    cur_n    = np.full(N, n_background)
    cur_mat  = [scene.background] * N         # indexed by global ray index

    for _ in range(MAX_DEPTH):
        if len(gidx) == 0:
            break

        # Only test active rays
        t_next, normals, mat_in, mat_out = scene.next_interface(
            o[gidx], d[gidx], [cur_mat[i] for i in gidx], t_min=1e-6
        )

        hit    = t_next < _INF
        no_hit = ~hit

        # ---- Rays that left the scene: apply NA cutoff ----
        if np.any(no_hit):
            nh_g = gidx[no_hit]
            cos_exit = np.einsum("ij,j->i", d[nh_g], opt_axis)
            intensity[nh_g[cos_exit < cos_na]] = 0.0

        if not np.any(hit):
            gidx = np.empty(0, dtype=np.intp)
            break

        h_g   = gidx[hit]      # global indices of hit rays
        h_loc = np.where(hit)[0]

        # Beer-Lambert
        mu = np.array([cur_mat[i].mu_optical for i in h_g])
        intensity[h_g] *= np.exp(-mu * t_next[hit])

        # Advance origins to interface
        new_orig = o[h_g] + t_next[hit, None] * d[h_g]

        # Snell's law
        n1    = cur_n[h_g]
        n2    = np.array([mat_out[j].n if mat_out[j] is not None
                          else n_background for j in h_loc])
        cos_i = np.abs(np.einsum("ij,ij->i", d[h_g], normals[hit]))
        intensity[h_g] *= _fresnel_T(n1, n2, cos_i)

        new_dirs = _snell_refract(d[h_g], normals[hit], n1, n2)
        tir = np.any(np.isnan(new_dirs), axis=1)
        intensity[h_g[tir]] = 0.0

        # Keep only successfully refracted rays
        ok   = ~tir
        ok_g = h_g[ok]
        o[ok_g]       = new_orig[ok]
        d[ok_g]       = new_dirs[ok]
        cur_n[ok_g]   = n2[ok]
        for j2, gi in enumerate(ok_g):
            cur_mat[gi] = mat_out[h_loc[ok][j2]]

        gidx = ok_g

    return intensity


# ---------------------------------------------------------------------------
# Public render function
# ---------------------------------------------------------------------------

def render(scene, goniometer, n_cond=1, jpeg_quality=85):
    """
    Render a bright-field microscope image of the scene at the current
    goniometer position.

    Parameters
    ----------
    scene      : Scene object (from scene.scene.load)
    goniometer : Goniometer object (current motor positions)
    n_cond     : int — number of condenser illumination rays per pixel
    jpeg_quality: int — JPEG compression quality

    Returns
    -------
    img_array : (H, W) float32 array, values in [0, 1]
    jpeg_bytes : bytes — JPEG-encoded grayscale image
    """
    cam = scene.camera_cfg
    W = int(cam.get("width",  640))
    H = int(cam.get("height", 480))
    pixel_size = float(cam.get("pixel_size", 0.005))   # mm at zoom=1
    zoom       = goniometer.zoom
    eff_px     = pixel_size / zoom

    na_obj  = float(cam.get("na_objective",  0.10))
    na_cond = float(cam.get("na_condenser",  0.07))

    g = scene.geometry
    fast       = np.array(g.get("camera_fast",  [1, 0, 0]), dtype=float)
    slow       = np.array(g.get("camera_slow",  [0, 1, 0]), dtype=float)
    opt_axis   = np.array(g.get("optical_axis", [0, 0,-1]), dtype=float)
    opt_axis  /= np.linalg.norm(opt_axis)

    # --- Build ray grid in lab frame ---
    px = (np.arange(W) - W / 2.0) * eff_px
    py = (np.arange(H) - H / 2.0) * eff_px
    gx, gy = np.meshgrid(px, py)   # (H, W)
    origins_lab = (gx[:, :, None] * fast + gy[:, :, None] * slow).reshape(-1, 3)
    # Shift origins so rays start well before the sample
    origins_lab -= opt_axis * 50.0   # 50 mm upstream

    # --- Inverse motor transform: move lab rays into sample frame ---
    T_inv = goniometer.transform_inv()
    origins_s = apply_transform(T_inv, origins_lab)
    base_dirs  = np.broadcast_to(opt_axis, (W * H, 3)).copy()
    dirs_s     = apply_transform_dirs(T_inv, base_dirs)

    # Optical axis in sample frame — needed for NA cutoff.
    # The camera is FIXED in the lab; only the sample rotates.  So the lab
    # optical axis transforms into the sample frame via T_inv's rotation part.
    opt_axis_s = T_inv[:3, :3] @ opt_axis
    opt_axis_s /= np.linalg.norm(opt_axis_s) + 1e-30

    # --- Condenser offsets ---
    offsets = _condenser_offsets(n_cond, na_cond)   # (n_cond, 3)

    accum = np.zeros(H * W)
    for k in range(n_cond):
        ox, oy, oz = offsets[k]
        # Tilt the illumination direction slightly
        illum_dir = opt_axis + ox * fast + oy * slow
        illum_dir /= np.linalg.norm(illum_dir)
        illum_dirs_s = apply_transform_dirs(T_inv,
                           np.broadcast_to(illum_dir, (W * H, 3)).copy())
        accum += _trace_rays(scene, origins_s, illum_dirs_s, na_obj,
                             opt_axis_sample=opt_axis_s)

    img = (accum / n_cond).reshape(H, W).astype(np.float32)

    # --- Encode JPEG ---
    img8 = (img * 255).clip(0, 255).astype(np.uint8)
    pil  = Image.fromarray(img8, mode="L")
    buf  = io.BytesIO()
    pil.save(buf, format="JPEG", quality=jpeg_quality)
    jpeg_bytes = buf.getvalue()

    return img, jpeg_bytes
