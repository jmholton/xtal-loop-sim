#!/usr/bin/env python3
"""
Attach a twisted-pair helix stem to a digitized hoop, producing a single
continuous loop fiber tube (stem + hoop + stem).

Terminology:
    hoop  — the oval/teardrop fiber part holding the crystal
    loop  — the complete assembly: stem + hoop

The stem helix axis is computed from the hoop waypoints (opposite to the
centroid of interior points).  The two helix strands and the hoop are joined
into one path so the entire loop is rendered as a single tube with uniform
capsule density:

    stem strand 1 (pin → junction)  +  hoop  +  stem strand 2 (junction → pin)

The cubic spline in the tube primitive bridges the ~fd/2 gap at each
junction smoothly.

Usage:
    python3 add_stem.py hoop.yaml [--output loop.yaml]
                        [--stem-length MM]      (default 0.7)
                        [--pitch-ratio N]       (default 20)
                        [--n-samples-per-mm N]  (default 100, i.e. 1 µm/capsule)

Input:  hoop.yaml — from digitize_fiber.py
Output: loop.yaml — objects: [one tube] + motor: {tx, ty}
                    ready to pass to generate_scene.py

Coordinate convention
---------------------
camera_slow = [0, -1, 0]  →  +y is UP in the image.
digitize_fiber.py stores dy = -(row - oy) * px_mm  (flipped), so waypoints
are already in the correct orientation.

Motor formula:
    tx = (origin_col - W/2) * px_mm
    ty = (origin_row - H/2) * px_mm   (NOT negated)
"""
import sys, os, argparse
sys.path.insert(0, '/home/jamesh/projects/loop_sim/claude')

import numpy as np
import yaml

from crystal_harvester.nylon_mechanics import helix_path


_DEFAULT_OUTPUT = os.path.join(os.path.dirname(__file__), 'loop.yaml')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('hoop',
                   help='Digitized hoop YAML (from digitize_fiber.py)')
    p.add_argument('--output', default=_DEFAULT_OUTPUT,
                   help=f'Output loop YAML (default: {_DEFAULT_OUTPUT})')
    p.add_argument('--stem-length', type=float, default=0.7,
                   help='Stem length in mm (default 0.7)')
    p.add_argument('--pitch-ratio', type=float, default=20.0,
                   help='Helix pitch / fiber_diameter ratio (default 20)')
    p.add_argument('--n-samples-per-mm', type=float, default=20.0,
                   help='Tube capsules per mm of arc length (default 20 = 50 µm/capsule)')
    return p.parse_args()


def _arc_length(pts):
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load digitized hoop
    # ------------------------------------------------------------------
    with open(args.hoop) as f:
        hoop_data = yaml.safe_load(f)

    px_mm  = float(hoop_data['pixel_size_um']) / 1000.0
    W      = int(hoop_data['image']['width'])
    H      = int(hoop_data['image']['height'])
    ox     = float(hoop_data['origin_px']['col'])
    oy     = float(hoop_data['origin_px']['row'])
    fd_mm  = float(hoop_data['fiber']['diameter_mm'])
    hoop_waypoints = hoop_data['fiber']['waypoints']   # [[x,y,z], ...]

    # Motor offset: map origin pixel to sample-frame translation
    tx = (ox - W / 2) * px_mm
    ty = (oy - H / 2) * px_mm   # NOT negated (camera_slow=[0,-1,0])

    # ------------------------------------------------------------------
    # Stem geometry
    # ------------------------------------------------------------------
    wp        = np.array(hoop_waypoints[1:-1], dtype=float)  # interior points
    centroid  = wp.mean(axis=0)
    stem_axis = -centroid / np.linalg.norm(centroid)          # away from hoop

    stem_length  = args.stem_length
    pitch        = args.pitch_ratio * fd_mm
    helix_radius = fd_mm / 2   # touching fibres: 2R = fd_mm

    stem1 = helix_path(helix_radius, pitch, stem_length, n_points=40,
                       phase_offset=0.0, axis=stem_axis)
    stem2 = helix_path(helix_radius, pitch, stem_length, n_points=40,
                       phase_offset=np.pi, axis=stem_axis)

    # helix_path winds in the plane perpendicular to stem_axis.  For a hoop
    # that lies in z=0, e1 = [0,0,1] so:
    #   stem1[0] ≈ [0, 0, +fd/2]   (phase 0:  +e1 side)
    #   stem2[0] ≈ [0, 0, -fd/2]   (phase π:  -e1 side)
    # The two strands are separated by one fiber diameter in z at the junction.
    # Adjust the hoop's first/last waypoints to sit at those junction positions
    # so the tube objects are geometrically adjacent with no gap.
    junc1 = stem1[0]   # exact junction point for strand 1
    junc2 = stem2[0]   # exact junction point for strand 2

    # ------------------------------------------------------------------
    # Three separate tube objects.  The stem is a twisted pair: two fibers
    # that physically cross each other periodically, producing crossover
    # shadows in the rendered image.  This requires two separate tube
    # objects (stem_1, stem_2) — a single continuous path cannot model
    # the crossing of one fiber over another.  The hoop gets its own tube
    # so it has an independent CubicSpline and full n_samples budget.
    # ------------------------------------------------------------------
    hoop_pts  = np.array(hoop_waypoints, dtype=float)
    # Separate the hoop crossover endpoints in z by one fiber diameter so
    # they meet the corresponding stem strands without a gap.
    hoop_pts[0]  = junc1
    hoop_pts[-1] = junc2
    hoop_arc  = _arc_length(hoop_pts)
    stem_arc  = _arc_length(stem1)   # same for stem2

    def n_for(arc):
        return max(50, int(arc * args.n_samples_per_mm) + 1)

    # ------------------------------------------------------------------
    # Output: partial scene — motor + objects only, no camera/materials
    # ------------------------------------------------------------------
    output = {
        'motor': {
            'tx': round(float(tx), 6),
            'ty': round(float(ty), 6),
        },
        'objects': [
            {
                'name':     'hoop',
                'material': 'nylon',
                'shape': {
                    'type':      'tube',
                    'diameter':  round(fd_mm, 6),
                    'path':      hoop_pts.tolist(),
                    'n_samples': n_for(hoop_arc),
                },
            },
            {
                'name':     'stem_1',
                'material': 'nylon',
                'shape': {
                    'type':      'tube',
                    'diameter':  round(fd_mm, 6),
                    'path':      stem1.tolist(),
                    'n_samples': n_for(stem_arc),
                },
            },
            {
                'name':     'stem_2',
                'material': 'nylon',
                'shape': {
                    'type':      'tube',
                    'diameter':  round(fd_mm, 6),
                    'path':      stem2.tolist(),
                    'n_samples': n_for(stem_arc),
                },
            },
        ],
    }

    with open(args.output, 'w') as f:
        yaml.dump(output, f, default_flow_style=None, sort_keys=False)

    ns_hoop = n_for(hoop_arc)
    ns_stem = n_for(stem_arc)
    print(f"Hoop:   {args.hoop}  arc={hoop_arc:.3f} mm  n_samples={ns_hoop}", file=sys.stderr)
    print(f"Stem:   length={stem_length} mm  pitch_ratio={args.pitch_ratio:.0f}"
          f"  arc={stem_arc:.3f} mm  n_samples={ns_stem}", file=sys.stderr)
    print(f"        axis=({stem_axis[0]:+.3f},{stem_axis[1]:+.3f},{stem_axis[2]:+.3f})",
          file=sys.stderr)
    print(f"Motor:  tx={tx:+.5f} mm  ty={ty:+.5f} mm", file=sys.stderr)
    print(f"Loop  → {args.output}", file=sys.stderr)
    print(f"Next:   python3 generate_scene.py {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
