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


def _spherical_cap_profile(R_loop, volume, n_z):
    """
    Return (r_profile, z_profile) for a spherical cap.

    r_profile[0]  = 0       (apex, r = 0)
    r_profile[-1] = R_loop  (rim)
    z_profile[0]  = h       (apex at top, toward +z / camera)
    z_profile[-1] = 0       (rim at z = 0, the loop plane)
    """
    V_hemi = (2.0 / 3.0) * np.pi * R_loop**3   # maximum achievable volume
    if volume >= V_hemi:
        if volume > V_hemi * 1.001:
            print(f"  WARNING: volume {volume:.6f} mm³ exceeds hemisphere "
                  f"({V_hemi:.6f} mm³); clamped.", file=sys.stderr)
        volume = V_hemi * 0.999   # a tiny amount below hemisphere

    # Solve V_cap(h) = volume for h in (0, R_loop]
    h_opt = brentq(lambda h: _cap_volume(h, R_loop) - volume,
                   1e-9, R_loop, xtol=1e-10, rtol=1e-10)

    rho = (R_loop**2 + h_opt**2) / (2.0 * h_opt)   # sphere radius

    # Parameterise by z' = depth from apex, running 0 → h_opt
    z_prime = np.linspace(0.0, h_opt, n_z)
    r_profile = np.sqrt(np.maximum(z_prime * (2.0 * rho - z_prime), 0.0))
    r_profile[0] = 0.0          # exact zero at apex

    # z_profile: apex at top (z = h_opt), rim at bottom (z = 0)
    z_profile = h_opt - z_prime

    return r_profile, z_profile, h_opt, rho


# ---------------------------------------------------------------------------
# Mesh revolution
# ---------------------------------------------------------------------------

def _revolve_profile(r_profile, z_profile, n_phi):
    """
    Revolve a 2-D meridional profile around the Z-axis.

    profile index 0  → apex   (r = 0, z = z_profile[0])
    profile index -1 → rim    (r = R_loop, z = z_profile[-1])

    Returns (vertices, faces).
    """
    n_z  = len(r_profile)
    phi  = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Main body vertices: (n_z, n_phi, 3)
    verts = np.zeros((n_z, n_phi, 3))
    for i in range(n_z):
        verts[i, :, 0] = r_profile[i] * cos_phi
        verts[i, :, 1] = r_profile[i] * sin_phi
        verts[i, :, 2] = z_profile[i]
    vertices = verts.reshape(-1, 3)

    # Body quads → 2 triangles each
    faces = []
    for i in range(n_z - 1):
        for j in range(n_phi):
            j1  = (j + 1) % n_phi
            v00 = i       * n_phi + j
            v01 = i       * n_phi + j1
            v10 = (i + 1) * n_phi + j
            v11 = (i + 1) * n_phi + j1
            faces.append([v00, v10, v01])
            faces.append([v10, v11, v01])

    # Apex cap: collapse first ring to a point
    apex_idx = len(vertices)
    vertices  = np.vstack([vertices, [[0.0, 0.0, z_profile[0]]]])
    for j in range(n_phi):
        j1 = (j + 1) % n_phi
        faces.append([apex_idx, j, j1])

    # Rim cap: flat disk closing the bottom (z = z_profile[-1])
    rim_start  = (n_z - 1) * n_phi
    center_idx = len(vertices)
    vertices   = np.vstack([vertices, [[0.0, 0.0, z_profile[-1]]]])
    for j in range(n_phi):
        j1 = (j + 1) % n_phi
        faces.append([center_idx, rim_start + j1, rim_start + j])

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

    # Interior waypoints: skip first and last (attachment / closing points)
    interior = np.array(hoop_waypoints[1:-1], dtype=float)
    centroid  = interior.mean(axis=0)   # (3,)

    # Effective loop radius: mean distance from centroid to interior waypoints
    dists  = np.linalg.norm(interior - centroid, axis=1)
    R_loop = float(dists.mean())

    print(f"Hoop centroid: ({centroid[0]:+.4f}, {centroid[1]:+.4f}, "
          f"{centroid[2]:+.4f}) mm", file=sys.stderr)
    print(f"R_loop (effective): {R_loop:.4f} mm", file=sys.stderr)
    print(f"V_hemisphere: {(2/3)*np.pi*R_loop**3:.6f} mm³", file=sys.stderr)

    # ------------------------------------------------------------------
    # Build spherical cap mesh
    # ------------------------------------------------------------------
    r_profile, z_profile, h_opt, rho = _spherical_cap_profile(
        R_loop, args.volume, args.n_z)

    vertices, faces = _revolve_profile(r_profile, z_profile, args.n_phi)

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

    print(f"Droplet: h={h_opt:.4f} mm  rho={rho:.4f} mm  "
          f"R_loop={R_loop:.4f} mm  vol={args.volume:.4f} mm³",
          file=sys.stderr)
    print(f"  Mesh: {len(vertices)} vertices, {len(faces)} faces",
          file=sys.stderr)
    print(f"Droplet → {args.output}", file=sys.stderr)
    print(f"Next:   python3 generate_scene.py loop.yaml droplet.yaml",
          file=sys.stderr)


if __name__ == '__main__':
    main()
