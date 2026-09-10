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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import yaml
from loop_sim.scene.tube import neville_sample
from crystal_harvester.droplet import (biconvex_lens_profile, loop_rim_radii,
                                       revolve_biconvex)


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
# Geometry lives in crystal_harvester.droplet (shared with the scene
# generator: one implementation, two callers).
# ---------------------------------------------------------------------------

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

    R_phi  = loop_rim_radii(centroid[:2], poly_xy, fiber_radius, phi_angles)
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
    r_profile, z_profile, h_opt, rho = biconvex_lens_profile(
        R_mean, args.volume, args.n_z, clamp=True)

    vertices, faces = revolve_biconvex(
        r_profile, z_profile, len(phi_angles),
        R_phi=R_phi, R_mean=R_mean, phi_angles=phi_angles)

    # Translate to loop centroid (loop lies roughly in z=0 plane)
    vertices = vertices + centroid

    # ------------------------------------------------------------------
    # Serialise to component YAML
    # ------------------------------------------------------------------
    # Round to 5 decimal places (0.01 µm precision, adequate for mm-scale scene)
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
