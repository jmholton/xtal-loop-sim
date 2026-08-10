"""
Goniometer pin geometry builder.

A goniometer pin is a metal cylinder (brass or stainless steel) with a
scored-and-broken beveled tip onto which the nylon loop is glued.

The geometry is: cylinder ∩ half_space (the scoring plane).

The pin extends along +Z (the same axis as the nylon stem), with the beveled
tip at z=0 and the pin body extending in the -Z direction.

Usage
-----
    from crystal_harvester.pin_geometry import make_pin

    shape_spec = make_pin(
        diameter_mm    = 0.5,
        length_mm      = 6.0,
        bevel_angle_deg = 45.0,
        bevel_offset_mm = 0.3,   # how far along from tip the score is
    )
"""

import numpy as np


_BEVEL_MAX_DEG = 80.0


def make_pin(diameter_mm, length_mm, bevel_angle_deg=45.0, bevel_offset_mm=0.3,
             tip_pos=(0.0, 0.0, 0.0), axis=(1.0, 0.0, 0.0), bevel_up=None):
    """
    Build a CSG intersection YAML spec for a beveled metal pin.

    The pin is a cylinder along `axis` starting at `tip_pos` (the beveled
    end nearest the sample) and extending length_mm in the axis direction.

    Parameters
    ----------
    diameter_mm     : float      — outer diameter of the pin cylinder
    length_mm       : float      — total visible length of the pin
    bevel_angle_deg : float      — tilt of scoring plane from perpendicular (0=flat cut)
    bevel_offset_mm : float      — axial distance from tip to the score
    tip_pos         : (3,) array — 3-D position of the pin tip (stem attachment end)
    axis            : (3,) array — unit vector pointing from tip into pin body

    Returns
    -------
    shape_spec : dict  — YAML-serialisable CSG intersection spec
    """
    tip = np.asarray(tip_pos, dtype=float)
    ax  = np.asarray(axis,    dtype=float)
    ax  = ax / np.linalg.norm(ax)

    radius = diameter_mm / 2.0
    center = tip + ax * (length_mm / 2.0)

    cylinder_spec = {
        "type":   "cylinder",
        "centre": [round(float(v), 6) for v in center.tolist()],
        "axis":   [round(float(v), 8) for v in ax.tolist()],
        "radius": round(radius, 6),
        "height": round(length_mm, 6),
    }

    # `bevel_angle_deg` is measured from perpendicular-to-axis, so 0 is a flat
    # cut and 90 does not cut the tip at all -- it slices the whole cylinder
    # lengthwise and leaves a half-pin staring up-beam.  Everything from ~78.6
    # up also pushes the stem's glue joint outside the metal (the stem is run
    # deliberately past the tip so it emerges from the score face).  Reachable
    # only since the `--pin-bevel 0` falsy-zero bug was fixed, hence the guard.
    if not (0.0 <= float(bevel_angle_deg) < _BEVEL_MAX_DEG):
        raise ValueError(
            f"bevel_angle_deg={bevel_angle_deg} is outside [0, {_BEVEL_MAX_DEG}); "
            "0 is a flat cut, and at 90 the half-space stops cutting the tip and "
            "bisects the pin along its length instead")

    # WHICH WAY the bevel faces is a real choice, and it used to be made by
    # accident: `up` defaulted to +Y and `cross(ax, up)` therefore landed on Z,
    # which is the camera's depth axis -- so the chisel presented in full
    # profile at spindle 90/270 and as a blunt taper at 0.  Nothing chose that.
    # (Note the guard below never fires for an X-aligned pin: dot is exactly 0.)
    up = np.array([0.0, 1.0, 0.0]) if bevel_up is None \
        else np.asarray(bevel_up, dtype=float)
    if abs(np.dot(ax, up)) > 0.9:
        up = np.array([0.0, 0.0, 1.0])
    perp = np.cross(ax, up);  perp /= np.linalg.norm(perp)

    # Outward normal of the bevel half-space (points toward tip/sample, away from pin body)
    theta  = np.deg2rad(bevel_angle_deg)
    outward = -np.cos(theta) * ax + np.sin(theta) * perp
    outward /= np.linalg.norm(outward)

    # HalfSpace: interior = {p : outward · p <= offset} = pin body side
    point_on_plane = tip + ax * bevel_offset_mm
    offset = float(np.dot(outward, point_on_plane))

    halfspace_spec = {
        "type":   "half_space",
        "normal": [round(float(v), 8) for v in outward.tolist()],
        "offset": round(offset, 8),
    }

    return {
        "type":     "intersection",
        "children": [cylinder_spec, halfspace_spec],
    }
