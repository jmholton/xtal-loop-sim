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

def _revolve_biconvex(r_profile, z_profile, n_phi):
    """
    Revolve a biconvex-lens meridional profile around the Z-axis.

    profile index 0       → top apex    (r = 0, z = +h)
    profile index n_z-1   → rim         (r = R_loop, z = 0)
    profile index 2*n_z-2 → bottom apex (r = 0, z = -h)

    Returns (vertices, faces) for a closed surface mesh.
    """
    n_pts = len(r_profile)
    phi     = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Main body vertices: (n_pts, n_phi, 3)
    verts = np.zeros((n_pts, n_phi, 3))
    for i in range(n_pts):
        verts[i, :, 0] = r_profile[i] * cos_phi
        verts[i, :, 1] = r_profile[i] * sin_phi
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

    # Interior waypoints: skip first and last (attachment / closing points)
    interior = np.array(hoop_waypoints[1:-1], dtype=float)
    centroid  = interior.mean(axis=0)   # (3,)

    # Effective loop radius: mean distance from centroid to interior waypoints,
    # minus one fiber radius so the droplet rim sits at the fiber inner surface
    # rather than the fiber axis.  This keeps the outer half of the fiber in air
    # and preserves normal bright-field contrast.
    dists  = np.linalg.norm(interior - centroid, axis=1)
    R_loop = float(dists.mean()) - fiber_radius

    print(f"Hoop centroid: ({centroid[0]:+.4f}, {centroid[1]:+.4f}, "
          f"{centroid[2]:+.4f}) mm", file=sys.stderr)
    print(f"R_loop (effective): {R_loop:.4f} mm", file=sys.stderr)
    print(f"V_hemisphere: {(2/3)*np.pi*R_loop**3:.6f} mm³", file=sys.stderr)

    # ------------------------------------------------------------------
    # Build biconvex lens mesh (symmetric about z=0, half-volume per side)
    # ------------------------------------------------------------------
    r_profile, z_profile, h_opt, rho = _biconvex_lens_profile(
        R_loop, args.volume, args.n_z)

    vertices, faces = _revolve_biconvex(r_profile, z_profile, args.n_phi)

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
          f"R_loop={R_loop:.4f} mm  vol={args.volume:.4f} mm³",
          file=sys.stderr)
    print(f"  Mesh: {len(vertices)} vertices, {len(faces)} faces",
          file=sys.stderr)
    print(f"Droplet → {args.output}", file=sys.stderr)
    print(f"Next:   python3 generate_scene.py loop.yaml droplet.yaml",
          file=sys.stderr)


if __name__ == '__main__':
    main()
