"""
Solvent droplet surface builder: closed-form spherical-cap (biconvex lens).

For cryo-crystallography loop sizes (~0.1-1.0 mm diameter) and water-like
solvents, Bo = rho*g*L^2/gamma ~ 0.003 (gravity negligible), so the
zero-gravity Young-Laplace solution for an axisymmetric droplet pinned on a
circular rim is *exactly* a spherical cap, the only axisymmetric
constant-mean-curvature surface.  A droplet wetting a loop bulges
symmetrically on both sides of the loop plane: two identical caps sharing
the rim circle, each holding half the volume.  The dome height h follows
analytically from the target volume:

    V/2 = pi * h * (3*R_loop^2 + h^2) / 6      (monotone in h, h in (0, R_loop])

With the contact line pinned at the loop rim, the contact angle is an
OUTPUT of (volume, rim radius), not an input: it cannot be prescribed
independently.  Replaces an earlier Bashforth-Adams ODE solve (see
docs/DECISIONS.md 2026-08-07, black droplet); there is no fallback path,
and impossible inputs raise.

Two public builders:

biconvex_lens_profile(R_loop, volume, n_z) -> r_profile, z_profile, h_opt, rho
    Meridional profile of the symmetric lens for a given rim radius and
    target volume.

droplet_in_loop(loop_pts, fiber_diameter_mm, volume_mm3, n_z, n_phi) ->
vertices, faces, info
    Full biconvex mesh pinned inside a loop's fiber waypoints: rim at the
    inner fiber edge, translated to the aperture centroid, straddling the
    loop plane.  info carries the measured h_mm/rho_mm/R_mean_mm/centroid/
    volume_centroid/volume_mm3.
"""

import sys

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Spherical-cap geometry
# ---------------------------------------------------------------------------

def cap_volume(h, R):
    """Volume of a spherical cap with dome height h and base radius R."""
    return np.pi * h * (3 * R**2 + h**2) / 6.0


def biconvex_lens_profile(R_loop, volume, n_z, clamp=False):
    """
    Return the meridional profile of a symmetric biconvex lens.

    The droplet wets the loop all the way around and bulges symmetrically on
    both sides of the loop plane (z=0): two identical spherical caps sharing
    the rim circle, each holding half the total volume.

    Parameters
    ----------
    R_loop : float — rim radius (mm)
    volume : float — total droplet volume (mm^3)
    n_z    : int   — samples per cap (profile has 2*n_z - 1 points)
    clamp  : bool  — a half-volume exceeding the hemisphere capacity is
             clamped with a warning instead of raising (legacy add_droplet
             behaviour; the generator uses the default and raises)

    Returns
    -------
    r_profile, z_profile : (2*n_z - 1,) arrays
        index 0        : top apex    (r = 0,      z = +h)
        index n_z - 1  : rim         (r = R_loop, z =  0)
        index 2*n_z-2  : bottom apex (r = 0,      z = -h)
    h_opt : float — dome height of each cap (mm)
    rho   : float — sphere radius of curvature (mm)
    """
    V_half = volume / 2.0
    V_hemi = (2.0 / 3.0) * np.pi * R_loop**3
    if V_half >= V_hemi:
        if not clamp:
            raise ValueError(
                f"droplet half-volume {V_half:.6f} mm^3 exceeds the hemisphere "
                f"capacity {V_hemi:.6f} mm^3 of a {R_loop:.4f} mm rim — the "
                f"loop cannot pin this much solvent; reduce --solvent-volume"
            )
        if V_half > V_hemi * 1.001:
            print(f"  WARNING: half-volume {V_half:.6f} mm³ exceeds hemisphere "
                  f"({V_hemi:.6f} mm³); clamped.", file=sys.stderr)
        V_half = V_hemi * 0.999

    h_opt = brentq(lambda h: cap_volume(h, R_loop) - V_half,
                   1e-9, R_loop, xtol=1e-10, rtol=1e-10)

    rho = (R_loop**2 + h_opt**2) / (2.0 * h_opt)   # sphere radius

    # Top-cap half: z' from 0 (apex) to h_opt (rim)
    z_prime = np.linspace(0.0, h_opt, n_z)
    r_half = np.sqrt(np.maximum(z_prime * (2.0 * rho - z_prime), 0.0))
    r_half[0] = 0.0
    z_half = h_opt - z_prime   # apex at +h, rim at 0

    # Full lens: top half + mirrored bottom half (skip shared rim)
    r_profile = np.concatenate([r_half, r_half[-2::-1]])    # (2*n_z-1,)
    z_profile = np.concatenate([z_half, -z_half[-2::-1]])   # mirrored z

    return r_profile, z_profile, h_opt, rho


def loop_rim_radii(centroid_xy, poly_xy, fiber_radius, phi_angles):
    """
    For each azimuthal angle, ray-cast from centroid to the loop polygon
    and return the effective rim radius (polygon distance minus fiber_radius).

    centroid_xy : (2,) center point
    poly_xy     : (N, 2) closed polygon vertices (fiber axis positions)
    phi_angles  : (n_phi,) array of angles in radians
    """
    cx, cy = centroid_xy
    poly = np.asarray(poly_xy, dtype=float)
    # Ensure closed
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])

    n_phi  = len(phi_angles)
    radii  = np.full(n_phi, np.inf)

    for k, phi in enumerate(phi_angles):
        dx, dy = np.cos(phi), np.sin(phi)
        for i in range(len(poly) - 1):
            ex = poly[i + 1, 0] - poly[i, 0]
            ey = poly[i + 1, 1] - poly[i, 1]
            denom = dx * ey - dy * ex
            if abs(denom) < 1e-12:
                continue
            rx = poly[i, 0] - cx
            ry = poly[i, 1] - cy
            t  = (rx * ey - ry * ex) / denom
            s  = (rx * dy - ry * dx) / denom
            if t > 1e-9 and -1e-9 <= s <= 1.0 + 1e-9:
                if t < radii[k]:
                    radii[k] = t

    # Fall back to a small positive value if no intersection found
    miss = ~np.isfinite(radii)
    if np.any(miss):
        print(f"  WARNING: {miss.sum()} phi directions missed the loop polygon; "
              "using fallback radius.", file=sys.stderr)
        radii[miss] = fiber_radius * 2.0

    return np.maximum(radii - fiber_radius, fiber_radius * 0.1)


# ---------------------------------------------------------------------------
# Mesh revolution
# ---------------------------------------------------------------------------

def revolve_biconvex(r_profile, z_profile, n_phi, R_phi=None, R_mean=None,
                     phi_angles=None):
    """
    Revolve a biconvex-lens meridional profile around the Z-axis.

    profile index 0       → top apex    (r = 0, z = +h)
    profile index n_z-1   → rim         (r = R_mean, z = 0)
    profile index 2*n_z-2 → bottom apex (r = 0, z = -h)

    The apex rows (r = 0) are NOT revolved into rings: each becomes a single
    vertex fanned to its neighbouring ring, so the mesh is watertight with no
    degenerate (zero-area) triangles.

    R_phi  : (n_phi,) per-angle rim radii.  If given, each phi column is
             scaled so the rim lands at R_phi[j] rather than R_mean, letting
             the droplet follow a non-circular loop outline.
    R_mean : scalar — the R_loop value used to build r_profile (the rim
             value in the profile).  Required when R_phi is given.

    Returns (vertices, faces) for a closed surface mesh.
    """
    r_profile = np.asarray(r_profile, dtype=float)
    z_profile = np.asarray(z_profile, dtype=float)
    if r_profile[0] != 0.0 or r_profile[-1] != 0.0:
        raise ValueError("profile must start and end on the axis (r = 0)")

    if phi_angles is not None:
        phi = np.asarray(phi_angles)
        n_phi = len(phi)
    else:
        phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Per-column scale factor: 1 everywhere if R_phi not supplied
    if R_phi is not None and R_mean is not None and R_mean > 0:
        scale = R_phi / R_mean          # (n_phi,)
    else:
        scale = np.ones(n_phi)

    # Ring rows: interior profile points only (indices 1 .. n_pts-2)
    r_rings = r_profile[1:-1]
    z_rings = z_profile[1:-1]
    n_rings = len(r_rings)

    verts = np.zeros((n_rings, n_phi, 3))
    for i in range(n_rings):
        verts[i, :, 0] = r_rings[i] * scale * cos_phi
        verts[i, :, 1] = r_rings[i] * scale * sin_phi
        verts[i, :, 2] = z_rings[i]
    vertices = verts.reshape(-1, 3)

    # Body quads → 2 triangles each
    faces = []
    for i in range(n_rings - 1):
        for j in range(n_phi):
            j1  = (j + 1) % n_phi
            v00 = i       * n_phi + j
            v01 = i       * n_phi + j1
            v10 = (i + 1) * n_phi + j
            v11 = (i + 1) * n_phi + j1
            faces.append([v00, v10, v01])
            faces.append([v10, v11, v01])

    # Top apex: fan to ring row 0
    apex_top = len(vertices)
    vertices = np.vstack([vertices, [[0.0, 0.0, z_profile[0]]]])
    for j in range(n_phi):
        j1 = (j + 1) % n_phi
        faces.append([apex_top, j, j1])

    # Bottom apex: fan to the last ring row, reversed winding
    apex_bot  = len(vertices)
    last_ring = (n_rings - 1) * n_phi
    vertices  = np.vstack([vertices, [[0.0, 0.0, z_profile[-1]]]])
    for j in range(n_phi):
        j1 = (j + 1) % n_phi
        faces.append([apex_bot, last_ring + j1, last_ring + j])

    return vertices, np.array(faces, dtype=int)


def mesh_volume_centroid(vertices, faces):
    """Signed volume and volume centroid of a closed triangle mesh.

    Divergence theorem: V = sum det(v0,v1,v2)/6, exact for a closed mesh and
    positive with consistent outward winding; wrong or inconsistent winding
    shows up as a wrong magnitude or sign, which is why the validator uses
    this rather than trusting the build.
    """
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))       # 6 * signed tetra vol
    volume = vol6.sum() / 6.0
    if abs(volume) < 1e-30:
        return 0.0, v.mean(axis=0)    # degenerate mesh; caller flags it
    centroid = (vol6[:, None] * (a + b + c)).sum(axis=0) / (24.0 * volume)
    return float(volume), centroid


def mesh_volume(vertices, faces):
    """Signed volume of a closed triangle mesh (see mesh_volume_centroid)."""
    return mesh_volume_centroid(vertices, faces)[0]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def droplet_in_loop(loop_pts, fiber_diameter_mm, volume_mm3,
                    n_z=30, n_phi=48, dense_poly_xy=None, phi_angles=None):
    """
    Build a biconvex droplet pinned inside a loop, placed at its aperture.

    The loop must lie in (or near) the z = 0 plane; build in that frame and
    transform the returned vertices together with the loop afterwards.

    Parameters
    ----------
    loop_pts          : (N, 3) fiber-axis waypoints of the loop
    fiber_diameter_mm : float — the rim pins at the inner fiber edge
    volume_mm3        : float — target droplet volume; raises ValueError if
                        the loop cannot pin it (no silent fallback)
    n_z, n_phi        : mesh resolution
    dense_poly_xy     : optional (M, 2) densely sampled fiber-axis polygon for
                        the rim ray-cast (the raw waypoints undershoot the
                        curved fiber between samples); defaults to loop_pts
    phi_angles        : optional explicit azimuth grid (radians); defaults to
                        n_phi uniform angles

    Returns
    -------
    vertices : (V, 3) float array — translated to the aperture centroid,
               straddling the loop plane, volume matching volume_mm3
    faces    : (F, 3) int array
    info     : dict — measured h_mm, rho_mm, R_mean_mm, centroid, volume_mm3
    """
    pts = np.asarray(loop_pts, dtype=float)
    if np.allclose(pts[0], pts[-1]):
        unique_pts = pts[:-1]
    else:
        unique_pts = pts

    # Centroid of the waypoints; for an asymmetric loop the crossover pulls
    # it toward the narrow end, which centres the apex in the aperture.
    centroid = unique_pts.mean(axis=0)   # (3,)

    if phi_angles is None:
        phi_angles = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    else:
        phi_angles = np.asarray(phi_angles, dtype=float)

    poly_xy = (np.asarray(dense_poly_xy, dtype=float)
               if dense_poly_xy is not None else unique_pts[:, :2])

    R_phi  = loop_rim_radii(centroid[:2], poly_xy, fiber_diameter_mm / 2.0,
                            phi_angles)
    R_mean = float(R_phi.mean())

    # Per-column scaling by R_phi/R_mean multiplies the enclosed volume by
    # mean((R_phi/R_mean)^2); solve the profile for a compensated volume so
    # the mesh comes out at the requested one.
    m2 = float(np.mean((R_phi / R_mean) ** 2))
    r_profile, z_profile, h_opt, rho = biconvex_lens_profile(
        R_mean, volume_mm3 / m2, n_z)

    vertices, faces = revolve_biconvex(
        r_profile, z_profile, len(phi_angles),
        R_phi=R_phi, R_mean=R_mean, phi_angles=phi_angles)

    # Exact-volume correction: faceting and the m2 approximation leave a
    # residual of a few percent; a pure z-scale fixes the volume without
    # moving the rim or the aperture footprint.
    v_now = mesh_volume(vertices, faces)
    if v_now <= 0:
        raise ValueError(
            f"droplet mesh has non-positive signed volume ({v_now:.3e} mm^3) "
            "— inconsistent face winding")
    vertices[:, 2] *= volume_mm3 / v_now
    h_opt *= volume_mm3 / v_now

    vertices = vertices + centroid

    # For an asymmetric aperture the drop's volume centroid is not the
    # waypoint centroid (more liquid sits toward the wide side); report it so
    # callers can centre a crystal in the actual solvent body.
    v_final, v_centroid = mesh_volume_centroid(vertices, faces)
    info = {
        "h_mm": float(h_opt),
        "rho_mm": float(rho),
        "R_mean_mm": R_mean,
        "centroid": centroid.tolist(),
        "volume_centroid": v_centroid.tolist(),
        "volume_mm3": v_final,
    }
    return vertices, faces, info
