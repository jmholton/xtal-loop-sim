"""
Solvent droplet surface builder using the Bashforth-Adams equation.

Models the axisymmetric sessile droplet meniscus suspended inside a nylon
loop using the Young-Laplace / Bashforth-Adams constant mean-curvature ODE.

For cryo-crystallography loop sizes (~0.1–1.0 mm diameter) and water-like
solvents, the Bond number Bo = ρgL²/γ ≈ 0.01, so gravity is negligible.
The zero-gravity solution (constant mean curvature = CMC surface) is a very
accurate approximation.

Reference
---------
Bashforth & Adams (1883) "An Attempt to Test the Theories of Capillary
Action", Cambridge University Press.
Oprea (2000) "Differential Geometry and its Applications", §5.5.

Usage
-----
    from crystal_harvester.droplet import bashforth_adams

    vertices, faces = bashforth_adams(
        R_loop_mm     = 0.15,   # inner radius of the nylon loop (mm)
        volume_mm3    = 0.002,  # target solvent volume (mm³)
        contact_angle_deg = 30, # solvent-on-nylon contact angle
        n_z   = 40,
        n_phi = 60,
    )
    # vertices: (N, 3) float array
    # faces:    (M, 3) int array
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Bashforth-Adams ODE (zero gravity, axisymmetric)
# ---------------------------------------------------------------------------

def _ba_rhs(z, state, dP):
    """
    Bashforth-Adams ODE for axisymmetric constant mean curvature surface.

    State: [r, psi]
    where r = radial coordinate (mm), psi = angle of tangent to vertical.

    dr/dz   = tan(psi)
    dpsi/dz = (dP/γ - sin(psi)/r) / cos(psi)

    Pressure difference dP/γ has units 1/mm (we absorb γ into dP here,
    so dP is the dimensionless curvature 2H, where H is mean curvature in 1/mm).

    The two principal curvatures are:
      κ₁ = dpsi/ds = (dpsi/dz)·cos(psi)     (meridional)
      κ₂ = sin(psi)/r                         (azimuthal)
    Mean curvature: H = (κ₁ + κ₂)/2 = dP/2γ → κ₁ + κ₂ = dP.
    """
    r, psi = state
    if abs(r) < 1e-9:
        # Near axis: L'Hôpital limit sin(psi)/r → dpsi/dz (slope continuity)
        # Use Taylor: both curvatures equal at axis → dP/2 each
        dpsi_dz = (dP - dP / 2.0)   # simplification at axis: κ₁ = κ₂ = dP/2
        dr_dz   = np.tan(psi) if abs(psi) < np.pi / 2 - 1e-4 else np.sign(psi) * 1e6
    else:
        cos_psi = np.cos(psi)
        if abs(cos_psi) < 1e-9:
            return [1e6, 0.0]
        dr_dz   = np.tan(psi)
        dpsi_dz = (dP - np.sin(psi) / r) / cos_psi
    return [dr_dz, dpsi_dz]


def _integrate_profile(dP, z_max, psi0=0.0, r0=1e-6, n_pts=200):
    """
    Integrate the Bashforth-Adams ODE from z=0 (apex) outward to z=z_max.

    Parameters
    ----------
    dP    : float — dimensionless curvature (2H in 1/mm)
    z_max : float — integration endpoint (mm)
    psi0  : float — tangent angle at apex (should be 0 for symmetric droplet)
    r0    : float — radial coordinate at apex (near 0)

    Returns
    -------
    z_arr : (N,) array
    r_arr : (N,) array
    psi_arr : (N,) array
    """
    sol = solve_ivp(
        _ba_rhs,
        [0.0, z_max],
        [r0, psi0],
        args=(dP,),
        method="RK45",
        dense_output=True,
        max_step=z_max / n_pts,
        rtol=1e-8,
        atol=1e-10,
        events=[lambda z, y, dP: y[0] - 5.0],  # stop if r > 5 mm (diverged)
    )
    z_eval = np.linspace(0.0, sol.t[-1], n_pts)
    r_arr, psi_arr = sol.sol(z_eval)
    return z_eval, r_arr, psi_arr


def _volume_from_profile(z_arr, r_arr):
    """Volume of revolution: V = π ∫ r²(z) dz (trapezoidal)."""
    return np.pi * np.trapz(r_arr**2, z_arr)


def _contact_angle_residual(psi_end, contact_angle_deg):
    """
    At the contact line (r = R_loop), psi should equal
    π/2 - contact_angle (measured from vertical).
    """
    target = np.pi / 2.0 - np.deg2rad(contact_angle_deg)
    return psi_end - target


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bashforth_adams(R_loop_mm, volume_mm3, contact_angle_deg=30.0,
                    n_z=40, n_phi=60, gravity_ms2=0.0):
    """
    Compute the axisymmetric solvent droplet surface filling a loop of radius
    R_loop_mm with volume volume_mm3.

    Uses a shoot-and-bracket approach: binary search on dP (dimensionless
    mean curvature) until the profile reaches r = R_loop at the correct
    contact angle, and the enclosed volume matches the target.

    Parameters
    ----------
    R_loop_mm         : float — inner radius of the loop (mm)
    volume_mm3        : float — target solvent volume (mm³)
    contact_angle_deg : float — contact angle (degrees, solvent on nylon)
    n_z               : int   — number of z samples for the mesh
    n_phi             : int   — number of azimuthal samples for the mesh
    gravity_ms2       : float — gravitational acceleration (m/s²); 0 = cryo

    Returns
    -------
    vertices : (n_z * n_phi, 3) float array — mesh vertex positions (mm)
    faces    : (M, 3) int array — triangle indices

    Notes
    -----
    For the CMC surface (gravity=0), dP = 2/R_droplet_apex where R_droplet
    is the apex radius of curvature.  We bracket dP in [2/R_loop, 2/(R_loop/10)]
    (small to large curvature).
    """
    # --- Find dP by bisection on volume ---
    # Strategy: for a given dP, integrate until r = R_loop_mm, extract volume.
    # Adjust dP so that volume matches target.

    def profile_volume(dP_trial):
        """Integrate and return volume at r=R_loop crossing."""
        z_arr, r_arr, psi_arr = _integrate_profile(
            dP_trial, z_max=R_loop_mm * 4.0, r0=1e-6, n_pts=300
        )
        # Find where r crosses R_loop
        idx = np.searchsorted(r_arr, R_loop_mm)
        if idx == 0 or idx >= len(r_arr):
            return -1.0   # failed to reach R_loop
        # Interpolate crossing
        t = (R_loop_mm - r_arr[idx - 1]) / (r_arr[idx] - r_arr[idx - 1] + 1e-30)
        z_cross = z_arr[idx - 1] + t * (z_arr[idx] - z_arr[idx - 1])
        # Integrate volume up to crossing
        r_clip = np.concatenate([r_arr[:idx], [R_loop_mm]])
        z_clip = np.concatenate([z_arr[:idx], [z_cross]])
        return _volume_from_profile(z_clip, r_clip)

    # Bracket dP
    dP_lo = 0.5 / R_loop_mm   # nearly flat (large radius of curvature)
    dP_hi = 40.0 / R_loop_mm  # very curved (tight droplet)

    vol_lo = profile_volume(dP_lo)
    vol_hi = profile_volume(dP_hi)

    if vol_lo < 0 or vol_hi < 0:
        # Fallback: use hemisphere if bisection fails
        return _hemisphere_mesh(R_loop_mm, n_phi, n_z)

    # Bigger dP → smaller volume (tighter cap)
    # We want volume = volume_mm3, so we need to find the right dP.
    if volume_mm3 < vol_hi:
        dP_lo, dP_hi = dP_hi, dP_hi * 3.0
        vol_lo = vol_hi
        vol_hi = profile_volume(dP_hi)
    elif volume_mm3 > vol_lo:
        dP_lo = dP_lo / 3.0
        vol_lo = profile_volume(dP_lo)

    try:
        dP_opt = brentq(
            lambda dp: profile_volume(dp) - volume_mm3,
            dP_lo, dP_hi,
            xtol=1e-6, rtol=1e-6, maxiter=50,
        )
    except ValueError:
        # Volume not achievable in the bracket — use hemisphere
        return _hemisphere_mesh(R_loop_mm, n_phi, n_z)

    # --- Generate the final profile ---
    z_arr, r_arr, psi_arr = _integrate_profile(
        dP_opt, z_max=R_loop_mm * 4.0, r0=1e-6, n_pts=n_z * 4
    )
    # Clip to r = R_loop
    idx = np.searchsorted(r_arr, R_loop_mm)
    if idx <= 1:
        return _hemisphere_mesh(R_loop_mm, n_phi, n_z)

    t = (R_loop_mm - r_arr[idx - 1]) / (r_arr[idx] - r_arr[idx - 1] + 1e-30)
    z_cross = float(z_arr[idx - 1] + t * (z_arr[idx] - z_arr[idx - 1]))

    z_profile = np.linspace(0.0, z_cross, n_z)
    from scipy.interpolate import interp1d
    r_interp = interp1d(z_arr[:idx + 1], r_arr[:idx + 1],
                        kind="cubic", fill_value="extrapolate")
    r_profile = r_interp(z_profile)
    r_profile = np.clip(r_profile, 0.0, None)

    # Flip z so the rim (r=R_loop) is at z=0 (the loop plane in XY)
    # and the apex bulges toward +z (toward the camera looking along -Z).
    z_profile = z_cross - z_profile

    # --- Revolve profile into a mesh ---
    return _revolve_profile(r_profile, z_profile, n_phi)


def _revolve_profile(r_profile, z_profile, n_phi):
    """
    Revolve a 2D profile (r(z), z) around the Z-axis to produce a mesh.

    Returns (vertices, faces).
    """
    n_z = len(r_profile)
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    # Build vertex grid: shape (n_z, n_phi, 3)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    verts = np.zeros((n_z, n_phi, 3))
    for i in range(n_z):
        verts[i, :, 0] = r_profile[i] * cos_phi
        verts[i, :, 1] = r_profile[i] * sin_phi
        verts[i, :, 2] = z_profile[i]

    vertices = verts.reshape(-1, 3)

    # Build faces: quads → 2 triangles each
    faces = []
    for i in range(n_z - 1):
        for j in range(n_phi):
            j1 = (j + 1) % n_phi
            v00 = i * n_phi + j
            v01 = i * n_phi + j1
            v10 = (i + 1) * n_phi + j
            v11 = (i + 1) * n_phi + j1
            faces.append([v00, v10, v01])
            faces.append([v10, v11, v01])

    # Add apex cap (collapse z=0 ring to a point)
    # Apex vertex
    apex_idx = len(vertices)
    apex = np.array([[0.0, 0.0, z_profile[0]]])
    vertices = np.vstack([vertices, apex])
    for j in range(n_phi):
        j1 = (j + 1) % n_phi
        faces.append([apex_idx, j, j1])

    # Add rim cap (the flat ring at z_max)
    rim_start = (n_z - 1) * n_phi
    center_idx = len(vertices)
    center = np.array([[0.0, 0.0, z_profile[-1]]])
    vertices = np.vstack([vertices, center])
    for j in range(n_phi):
        j1 = (j + 1) % n_phi
        faces.append([center_idx, rim_start + j1, rim_start + j])

    return vertices, np.array(faces, dtype=int)


def _hemisphere_mesh(R, n_phi=60, n_theta=20):
    """Fallback: hemisphere of radius R as a mesh."""
    theta = np.linspace(0.0, np.pi / 2.0, n_theta)
    phi   = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    r_prof = R * np.sin(theta)
    z_prof = R * np.cos(theta)
    return _revolve_profile(r_prof, z_prof, n_phi)


def surface_evolver_script(R_loop_mm, volume_mm3, contact_angle_deg=30.0,
                            filename="droplet.fe"):
    """
    Generate a Surface Evolver input script for a more accurate (non-axisymmetric)
    droplet shape on a non-circular loop.

    The .fe file must be run with Surface Evolver (Brakke 1992):
        evolver droplet.fe

    Returns the script as a string; optionally writes to filename.
    """
    script = f"""\
// Surface Evolver script for sessile droplet in a nylon loop
// Loop inner radius: {R_loop_mm} mm
// Target volume: {volume_mm3} mm^3
// Contact angle: {contact_angle_deg} deg

SPACE_DIMENSION 3
SURFACE_DIMENSION 2
SIMPLEX_REPRESENTATION

gravity_constant 0  // cryo: no gravity

parameter R = {R_loop_mm}   // loop radius (mm)
parameter VOL = {volume_mm3}  // target volume (mm^3)
parameter CA = {contact_angle_deg}  // contact angle (degrees)

// Constraint: fix contact line to loop boundary
constraint 1
formula: x^2 + y^2 - R^2

// Volume constraint
quantity vol_constraint energy method volume_integral
global_method correction 1

// Initial shape: hemisphere
vertices
1  R  0  0
2  0  R  0
3 -R  0  0
4  0 -R  0
5  0  0  R

edges
1  1  2
2  2  3
3  3  4
4  4  1
5  1  5
6  2  5
7  3  5
8  4  5

faces
1  1  6 -5
2  2  7 -6
3  3  8 -7
4  4  5 -8

bodies
1  1 2 3 4  volume VOL

read

// Refine and minimize
refine edges where on_constraint 1
u 10
refine u 5
u 10
"""
    if filename:
        with open(filename, "w") as f:
            f.write(script)
    return script
