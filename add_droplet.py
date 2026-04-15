#!/usr/bin/env python3
"""
Compute a solvent droplet (spherical cap) inside a digitized hoop and write it
as a surface_mesh component YAML ready for generate_scene.py.

Physics
-------
For cryo-crystallography loop sizes the Bond number Bo = ρgL²/γ ≈ 0.01, so
gravity is negligible.  The zero-gravity Young-Laplace solution for an
axisymmetric droplet is exactly a spherical cap: the only constant-mean-
curvature axisymmetric surface.

Geometry
--------
The cap sits inside the loop with:
  • rim  at z = 0  (loop plane)
  • apex at z = h  (dome height, toward the camera along +z)

The rim radius R_loop is the mean distance from the loop interior centroid
to the digitized waypoints.  Dome height h is found analytically from the
target volume:

    V = π·h·(3·R_loop² + h²) / 6       (monotone in h; h ∈ (0, R_loop])

The mesh is then translated to the loop interior centroid (cx, cy, 0).

Usage
-----
    python3 add_droplet.py hoop.yaml [--output droplet.yaml]
                           [--volume MM3]          (default 0.001)
                           [--n-z N]               (default 16)
                           [--n-phi N]             (default 32)

Output is a component YAML (objects only) for generate_scene.py.

Full pipeline
-------------
    python3 digitize_fiber.py real_loop.jpg --output hoop.yaml
    python3 add_stem.py   hoop.yaml              # → loop.yaml
    python3 add_droplet.py hoop.yaml             # → droplet.yaml
    python3 generate_scene.py loop.yaml droplet.yaml
    python3 render.py scene.yaml
"""
import sys, os, argparse
sys.path.insert(0, '/home/jamesh/projects/loop_sim/claude')

import numpy as np
import yaml
from scipy.optimize import brentq
from loop_sim.scene.tube import neville_sample


_DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), 'droplet.yaml')


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('hoop',
                   help='Digitized hoop YAML (from digitize_fiber.py)')
    p.add_argument('--output', default=_DEFAULT_OUTPUT,
                   help=f'Output component YAML (default: {_DEFAULT_OUTPUT})')
    p.add_argument('--volume', type=float, default=0.001,
                   help='Target solvent volume in mm³ (default: 0.001 = 1 nL)')
    p.add_argument('--n-z',   type=int,   default=16,
                   help='Meridional rings in the mesh (default: 16)')
    p.add_argument('--n-phi', type=int,   default=32,
                   help='Azimuthal segments in the mesh (default: 32)')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Spherical cap geometry
# ---------------------------------------------------------------------------

def _cap_volume(h, R):
    """Volume of spherical cap with dome height h and base radius R."""
    return np.pi * h * (3 * R**2 + h**2) / 6.0


def _loop_rim_radii(centroid_xy, poly_xy, fiber_radius, phi_angles):
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


def _biconvex_lens_profile(R_loop, volume, n_z):
    """
    Return (r_profile, z_profile) for a symmetric biconvex lens.

    The droplet wets the nylon all the way around the hoop and bulges
    symmetrically on both sides of the loop plane (z=0).  With gravity
    negligible (Bond number << 1), the equilibrium shape is two identical
    spherical caps sharing the rim circle.  Each cap holds half the total
    volume.

    Returns arrays of length 2*n_z - 1:
      index 0        : top apex    (r = 0,      z = +h)
      index n_z - 1  : rim         (r = R_loop, z =  0)
      index 2*n_z-2  : bottom apex (r = 0,      z = -h)
    """
    V_half = volume / 2.0
    V_hemi = (2.0 / 3.0) * np.pi * R_loop**3
    if V_half >= V_hemi:
        if V_half > V_hemi * 1.001:
            print(f"  WARNING: half-volume {V_half:.6f} mm³ exceeds hemisphere "
                  f"({V_hemi:.6f} mm³); clamped.", file=sys.stderr)
        V_half = V_hemi * 0.999

    h_opt = brentq(lambda h: _cap_volume(h, R_loop) - V_half,
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


# ---------------------------------------------------------------------------
# Mesh revolution
# ---------------------------------------------------------------------------

def _revolve_biconvex(r_profile, z_profile, n_phi, R_phi=None, R_mean=None,
                      phi_angles=None):
    """
    Revolve a biconvex-lens meridional profile around the Z-axis.

    profile index 0       → top apex    (r = 0, z = +h)
    profile index n_z-1   → rim         (r = R_mean, z = 0)
    profile index 2*n_z-2 → bottom apex (r = 0, z = -h)

    R_phi  : (n_phi,) per-angle rim radii.  If given, each phi column is
             scaled so the rim lands at R_phi[j] rather than R_mean, letting
             the droplet follow a non-circular loop outline.
    R_mean : scalar — the R_loop value used to build r_profile (the rim
             value in the profile).  Required when R_phi is given.

    Returns (vertices, faces) for a closed surface mesh.
    """
    n_pts = len(r_profile)
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

    # Main body vertices: (n_pts, n_phi, 3)
    verts = np.zeros((n_pts, n_phi, 3))
    for i in range(n_pts):
        verts[i, :, 0] = r_profile[i] * scale * cos_phi
        verts[i, :, 1] = r_profile[i] * scale * sin_phi
        verts[i, :, 2] = z_profile[i]
    vertices = verts.reshape(-1, 3)

    # Body quads → 2 triangles each (includes degenerate quads at apices,
    # which are zero-area and harmless)
    faces = []
    for i in range(n_pts - 1):
        for j in range(n_phi):
            j1  = (j + 1) % n_phi
            v00 = i       * n_phi + j
            v01 = i       * n_phi + j1
            v10 = (i + 1) * n_phi + j
            v11 = (i + 1) * n_phi + j1
            faces.append([v00, v10, v01])
            faces.append([v10, v11, v01])

    # Top apex cap: fan from separate apex vertex to row 0 (degenerate ring)
    apex_top = len(vertices)
    vertices  = np.vstack([vertices, [[0.0, 0.0, z_profile[0]]]])
    for j in range(n_phi):
        j1 = (j + 1) % n_phi
        faces.append([apex_top, j, j1])

    # Bottom apex cap: fan from separate apex vertex to last row, reversed winding
    apex_bot  = len(vertices)
    last_ring = (n_pts - 1) * n_phi
    vertices  = np.vstack([vertices, [[0.0, 0.0, z_profile[-1]]]])
    for j in range(n_phi):
        j1 = (j + 1) % n_phi
        faces.append([apex_bot, last_ring + j1, last_ring + j])

    return vertices, np.array(faces, dtype=int)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load hoop data
    # ------------------------------------------------------------------
    with open(args.hoop) as f:
        hoop_data = yaml.safe_load(f)

    hoop_waypoints = hoop_data['fiber']['waypoints']
    fiber_radius   = float(hoop_data['fiber'].get('diameter_mm', 0.0)) / 2.0

    # All unique waypoints (first == last for a closed loop; deduplicate)
    all_pts  = np.array(hoop_waypoints, dtype=float)
    if np.allclose(all_pts[0], all_pts[-1]):
        unique_pts = all_pts[:-1]
        closed_pts = all_pts           # already has repeated endpoint
    else:
        unique_pts = all_pts
        closed_pts = np.vstack([all_pts, all_pts[0:1]])   # close the loop

    # Centroid from all unique waypoints so the crossover pulls it toward
    # the narrow end and the apex sits more centrally in the loop.
    centroid = unique_pts.mean(axis=0)   # (3,)

    # Dense-sample the hoop using the same CubicSpline as the tube renderer
    # so the droplet rim follows the actual smooth fiber path, not just the
    # straight-edge polygon between the sparse waypoints.  Between waypoints
    # the fiber curves outward; using only the raw waypoints undershoots the
    # rim radius and leaves a visible gap between the droplet edge and the fiber.
    n_dense   = max(100, 10 * len(closed_pts))
    dense_pts = neville_sample(closed_pts, n_dense)   # (n_dense, 3)
    # Drop the last sample (= first, since the loop is closed) to get a proper
    # open polygon that _loop_rim_radii will close automatically.
    poly_xy   = dense_pts[:-1, :2]   # (n_dense-1, 2)
    cx2, cy2  = centroid[0], centroid[1]

    # Base uniform phi grid
    phi_uniform = np.linspace(0.0, 2.0 * np.pi, args.n_phi, endpoint=False)

    # Add exact angles toward every original waypoint vertex so the rim mesh
    # is guaranteed to reach each corner (especially the crossover at [0,0]).
    vertex_phis = np.arctan2(unique_pts[:, 1] - cy2,
                             unique_pts[:, 0] - cx2) % (2 * np.pi)
    phi_angles  = np.unique(np.concatenate([phi_uniform, vertex_phis]))

    R_phi  = _loop_rim_radii(centroid[:2], poly_xy, fiber_radius, phi_angles)
    R_mean = float(R_phi.mean())

    print(f"Hoop centroid: ({centroid[0]:+.4f}, {centroid[1]:+.4f}, "
          f"{centroid[2]:+.4f}) mm", file=sys.stderr)
    print(f"R_loop (mean effective): {R_mean:.4f} mm  "
          f"min={R_phi.min():.4f}  max={R_phi.max():.4f}", file=sys.stderr)
    print(f"V_hemisphere (mean): {(2/3)*np.pi*R_mean**3:.6f} mm³", file=sys.stderr)

    # ------------------------------------------------------------------
    # Build biconvex lens mesh (symmetric about z=0, half-volume per side)
    # h_opt is solved for R_mean; each phi column is then scaled by R_phi/R_mean
    # so the rim follows the actual (non-circular) loop outline.
    # ------------------------------------------------------------------
    r_profile, z_profile, h_opt, rho = _biconvex_lens_profile(
        R_mean, args.volume, args.n_z)

    vertices, faces = _revolve_biconvex(
        r_profile, z_profile, len(phi_angles),
        R_phi=R_phi, R_mean=R_mean, phi_angles=phi_angles)

    # Translate to loop centroid (loop lies roughly in z=0 plane)
    vertices = vertices + centroid

    # ------------------------------------------------------------------
    # Serialise to component YAML
    # ------------------------------------------------------------------
    # Round to 5 decimal places (0.01 µm precision — adequate for mm-scale scene)
    verts_list = [[round(float(v), 5) for v in row] for row in vertices]
    faces_list = [[int(f) for f in row] for row in faces]

    output = {
        'objects': [
            {
                'name':     'droplet',
                'material': 'solvent',
                'shape': {
                    'type':     'surface_mesh',
                    'vertices': verts_list,
                    'faces':    faces_list,
                },
            }
        ]
    }

    with open(args.output, 'w') as f:
        yaml.dump(output, f, default_flow_style=None, sort_keys=False)

    print(f"Droplet: h={h_opt:.4f} mm (each side)  rho={rho:.4f} mm  "
          f"R_mean={R_mean:.4f} mm  vol={args.volume:.4f} mm³",
          file=sys.stderr)
    print(f"  Mesh: {len(vertices)} vertices, {len(faces)} faces",
          file=sys.stderr)
    print(f"Droplet → {args.output}", file=sys.stderr)
    print(f"Next:   python3 generate_scene.py loop.yaml droplet.yaml",
          file=sys.stderr)


if __name__ == '__main__':
    main()
