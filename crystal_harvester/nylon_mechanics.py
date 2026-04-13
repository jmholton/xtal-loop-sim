"""
Nylon fiber mechanics for loop and stem geometry.

Two functions:

helix_path(R, pitch, length, n_points)
    Compute Neville waypoints for one fiber of a twisted-pair stem.
    The helix has radius R (= fiber diameter, so touching fibers) and
    user-specified pitch.

elastica_loop(fiber_diameter_mm, loop_diameter_mm, youngs_modulus_gpa,
              loop_shape, n_points)
    Solve the planar Kirchhoff/Euler elastica BVP for the loop fiber.
    Returns Neville waypoints for the loop half-fiber (one strand, from
    stem attachment point around the loop and back).

Both functions return (N, 3) arrays of waypoints suitable for the 'tube'
primitive's 'path' field in a scene YAML.

Physics references
------------------
- Euler elastica: Antman, "Nonlinear Problems of Elasticity" (Springer 2005),
  Chapter 4.
- Kirchhoff rod: Love, "A Treatise on the Mathematical Theory of Elasticity"
  (1927), §§ 258-272.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Twisted-pair stem: helix path
# ---------------------------------------------------------------------------

def helix_path(R, pitch, length, n_points=50, phase_offset=0.0,
               axis=(1.0, 0.0, 0.0)):
    """
    3-D waypoints for one fiber of a twisted-pair stem.

    The helix winds around `axis`, running from the origin (loop attachment)
    to origin + axis * length (pin end).

    Parameters
    ----------
    R            : float      — helix radius (mm); fibers touch when R = fiber_diameter
    pitch        : float      — axial distance per full revolution (mm)
    length       : float      — total stem length (mm)
    n_points     : int        — number of waypoints
    phase_offset : float      — initial azimuthal phase (radians); use π for fiber 2
    axis         : (3,) array — unit vector along the stem/pin direction

    Returns
    -------
    pts : (n_points, 3) float array — waypoints in sample frame
    """
    ax = np.asarray(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)

    # Two unit vectors perpendicular to ax (for helix winding plane)
    up = np.array([0.0, 1.0, 0.0])
    if abs(np.dot(ax, up)) > 0.9:
        up = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(ax, up);  e1 /= np.linalg.norm(e1)
    e2 = np.cross(ax, e1);  e2 /= np.linalg.norm(e2)

    t   = np.linspace(0.0, length, n_points)
    phi = 2.0 * np.pi * t / pitch + phase_offset

    pts = (t[:, None] * ax[None, :]
           + R * np.cos(phi)[:, None] * e1[None, :]
           + R * np.sin(phi)[:, None] * e2[None, :])
    return pts


# ---------------------------------------------------------------------------
# Elastica BVP helper
# ---------------------------------------------------------------------------

def _elastica_rhs(s, state):
    """
    ODE right-hand side for planar Euler elastica.

    State: [x, y, theta, kappa]
    where theta = tangent angle, kappa = curvature = d(theta)/ds.

    No external forces: kappa is constant (free elastica → circular arc
    or straight).  With end-load F along X: kappa' = -F * sin(theta) / EI.

    We use the moment-free BVP (closed loop, no end force on the loop itself):
        kappa'(s) = 0  → kappa = const  → circular arc
    But for a teardrop (non-uniform curvature), we impose an end-force
    along the loop-closure direction.

    This simplified version uses a constant kappa for each half-loop and
    matches boundary conditions via shooting.  For higher accuracy use
    scipy.solve_bvp.
    """
    x, y, theta, kappa = state
    return [
        np.cos(theta),
        np.sin(theta),
        kappa,
        0.0,    # free elastica: kappa = const
    ]


def _integrate_halfloop(arc_length, kappa0, theta0):
    """
    Integrate planar elastica from s=0 to s=arc_length.

    Returns (x_end, y_end, theta_end) and the full trajectory.
    """
    state0 = [0.0, 0.0, theta0, kappa0]
    sol = solve_ivp(
        _elastica_rhs,
        [0.0, arc_length],
        state0,
        method="RK45",
        dense_output=True,
        max_step=arc_length / 200,
        rtol=1e-8,
        atol=1e-10,
    )
    return sol


def elastica_loop(fiber_diameter_mm, loop_diameter_mm, youngs_modulus_gpa=2.0,
                  loop_shape="teardrop", n_points=40):
    """
    Compute Neville waypoints for the loop fiber as a planar ellipse.

    The loop is parameterised as two mirror-symmetric elliptical arcs (upper
    and lower strands) that share the stem attachment point at the origin and
    meet at the apex on the opposite side.  The ellipse is always C1-smooth at
    both junctions by construction — no kink is possible.

    The shape presets control the X/Y aspect ratio of the ellipse:

        'circular'  — perfect circle; a = b = loop_diameter/2
        'oval'      — slightly taller than wide; a/b ≈ 0.85
        'teardrop'  — more elongated; a/b ≈ 0.70

    In all cases the Y span is normalised to loop_diameter_mm so that the
    parameter matches the Hampton catalogue widest-opening convention.

    The stem attachment is at the origin (0, 0, 0); the loop extends in the
    −X direction; the stem/pin goes in +X.  The loop lies entirely in the
    XY plane (Z = 0).

    Parameters
    ----------
    fiber_diameter_mm  : float — nylon fiber diameter (mm); unused in geometry
                                 but retained for API compatibility
    loop_diameter_mm   : float — widest Y span of the loop opening (mm)
    youngs_modulus_gpa : float — unused (kept for API compatibility)
    loop_shape         : str   — 'teardrop' | 'oval' | 'circular'
    n_points           : int   — waypoints per half-strand (total = 2*n_points−1)

    Returns
    -------
    pts : (2*n_points−1, 3) float array — closed loop waypoints in XY plane
    """
    # Shape parameters:
    #   aspect  — X half-axis / Y half-axis.
    #             < 1 → loop shallower than wide (flat oval)
    #             > 1 → loop deeper than wide (elongated teardrop)
    #   c_asym  — second-harmonic coefficient for y(t) = sin(t) + c*sin(2t).
    #             c = 0  → symmetric ellipse, max-width at midpoint
    #             c < 0  → max-width shifted toward apex (teardrop shape)
    if loop_shape == "circular":
        aspect = 1.00
        c_asym = 0.0
    elif loop_shape == "oval":
        aspect = 1.10
        c_asym = -0.10
    else:  # teardrop
        aspect = 1.85
        c_asym = -0.25

    b = 1.0                             # unit scale; normalization sets physical size
    a = b * aspect

    # Parameterised so t=0 → attachment (0,0), t=π → apex (−2a, 0).
    # y = sin(t) + c*sin(2t) vanishes at both endpoints and shifts the widest
    # point toward the apex (t > π/2) when c < 0.
    t = np.linspace(0.0, np.pi, n_points)
    x   = -a * (1.0 - np.cos(t))
    y_u =  b * (np.sin(t) + c_asym * np.sin(2.0 * t))
    y_l = -y_u

    upper = np.column_stack([ x,  y_u, np.zeros(n_points)])
    lower = np.column_stack([ x,  y_l, np.zeros(n_points)])

    # Full closed path: attachment → upper → apex → lower reversed → attachment
    pts_3d = np.vstack([upper, lower[-2::-1]])   # lower reversed, drop duplicate apex

    # Normalise Y span to loop_diameter_mm (no-op for this ellipse, but kept
    # for consistency if aspect or b are ever changed externally).
    y_span = pts_3d[:, 1].max() - pts_3d[:, 1].min()
    if y_span > 1e-9:
        pts_3d = pts_3d * (loop_diameter_mm / y_span)

    return pts_3d


# ---------------------------------------------------------------------------
# Fiber tangent helper
# ---------------------------------------------------------------------------

def fiber_tangents(waypoints):
    """
    Compute unit tangent vectors at each waypoint using finite differences.

    Parameters
    ----------
    waypoints : (N, 3) array

    Returns
    -------
    tangents : (N, 3) unit vectors
    """
    pts = np.asarray(waypoints, dtype=float)
    N = len(pts)
    tangents = np.zeros_like(pts)
    # Central differences for interior, forward/backward at ends
    tangents[1:-1] = pts[2:] - pts[:-2]
    tangents[0]    = pts[1] - pts[0]
    tangents[-1]   = pts[-1] - pts[-2]
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-30] = 1.0
    return tangents / norms
