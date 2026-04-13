"""
Crystal shape builder.

Generates CSG half-space intersection nodes for standard protein crystal habits.
Face normals are derived from the crystal lattice orientation using reciprocal
lattice vectors (XDS convention).

Usage
-----
    from crystal_harvester.crystal_shapes import make_crystal

    # XDS-style real-space unit cell vectors in Angstroms
    lattice = {
        "a_axis": [10.5,  0.0,  0.0],
        "b_axis": [ 0.0, 10.5,  0.0],
        "c_axis": [ 0.0,  0.0, 27.3],
    }

    shape_spec, lattice_meta = make_crystal(
        preset="hexagonal",
        dimensions_mm=[0.08, 0.04],   # [width, height] or [a, b, c]
        lattice_abc=lattice,
    )
"""

import numpy as np

# ---------------------------------------------------------------------------
# Reciprocal lattice
# ---------------------------------------------------------------------------

def _reciprocal_vectors(a, b, c):
    """
    Compute reciprocal lattice vectors a*, b*, c* from real-space vectors a, b, c.

    a* = (b × c) / V,  b* = (c × a) / V,  c* = (a × b) / V
    where V = a · (b × c).

    All inputs in Angstroms (or any consistent unit); outputs are in 1/unit.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    V = np.dot(a, np.cross(b, c))
    if abs(V) < 1e-30:
        raise ValueError("Degenerate unit cell: volume is zero.")
    a_star = np.cross(b, c) / V
    b_star = np.cross(c, a) / V
    c_star = np.cross(a, b) / V
    return a_star, b_star, c_star


def _face_normal(hkl, a_star, b_star, c_star):
    """
    Outward face normal for Miller plane (h k l) in lab frame.

    n_hkl = h·a* + k·b* + l·c*  (unnormalised)
    """
    h, k, l = hkl
    n = h * a_star + k * b_star + l * c_star
    length = np.linalg.norm(n)
    if length < 1e-30:
        raise ValueError(f"Zero normal for hkl={hkl}.")
    return n / length


def _halfspace_spec(normal, offset_mm):
    """Return a half-space YAML spec dict."""
    return {
        "type":   "half_space",
        "normal": [round(float(v), 8) for v in normal],
        "offset": round(float(offset_mm), 8),
    }


# ---------------------------------------------------------------------------
# Presets — (Miller index, dimension key) pairs
# ---------------------------------------------------------------------------

# Each preset is a list of (hkl, dimension_index) tuples.
# dimension_index selects which element of `dimensions_mm` gives the half-width
# for that face pair.  Convention: dimension_mm[i] is the half-width (distance
# from origin to face) in the direction of that face normal.

_PRESETS = {
    # 6 faces: ±a, ±b, ±c
    "cube": [
        ((1, 0, 0), 0), ((-1, 0, 0), 0),
        ((0, 1, 0), 1), ((0,-1, 0), 1),
        ((0, 0, 1), 2), ((0, 0,-1), 2),
    ],
    # thin along c, wide in ab  — dimensions_mm: [ab_half, c_half]
    "plate": [
        ((1, 0, 0), 0), ((-1, 0, 0), 0),
        ((0, 1, 0), 0), ((0,-1, 0), 0),
        ((0, 0, 1), 1), ((0, 0,-1), 1),
    ],
    # elongated along c — dimensions_mm: [ab_half, c_half]
    "needle": [
        ((1, 0, 0), 0), ((-1, 0, 0), 0),
        ((0, 1, 0), 0), ((0,-1, 0), 0),
        ((1, 1, 0), 0), ((-1,-1, 0), 0),
        ((1,-1, 0), 0), ((-1, 1, 0), 0),
        ((0, 0, 1), 1), ((0, 0,-1), 1),
    ],
    # hexagonal prism — dimensions_mm: [r_half, c_half]
    "hexagonal": [
        ((1, 0, 0),    0), ((-1, 0, 0),   0),
        ((0, 1, 0),    0), ((0, -1, 0),   0),
        ((1,-1, 0),    0), ((-1, 1, 0),   0),
        ((0, 0, 1),    1), ((0, 0,-1),    1),
    ],
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_crystal(preset, dimensions_mm, lattice_abc):
    """
    Build a CSG intersection YAML spec for a crystal of the given habit.

    Parameters
    ----------
    preset : str
        One of 'cube', 'plate', 'needle', 'hexagonal'.
    dimensions_mm : list of float
        Half-widths in mm.
        - cube:       [a_half, b_half, c_half]
        - plate:      [ab_half, c_half]
        - needle:     [ab_half, c_half]
        - hexagonal:  [r_half, c_half]
    lattice_abc : dict with keys 'a_axis', 'b_axis', 'c_axis'
        Real-space unit cell vectors in Angstroms (XDS convention).

    Returns
    -------
    shape_spec : dict
        YAML-serialisable CSG intersection spec.
    lattice_meta : dict
        Lattice metadata to attach to the scene object (a/b/c axes in Å).
    """
    if preset not in _PRESETS:
        raise ValueError(f"Unknown crystal preset {preset!r}. "
                         f"Choose from {list(_PRESETS)}")

    a = np.asarray(lattice_abc["a_axis"], dtype=float)
    b = np.asarray(lattice_abc["b_axis"], dtype=float)
    c = np.asarray(lattice_abc["c_axis"], dtype=float)
    a_star, b_star, c_star = _reciprocal_vectors(a, b, c)

    dims = list(dimensions_mm)
    children = []
    for hkl, dim_idx in _PRESETS[preset]:
        if dim_idx >= len(dims):
            raise ValueError(
                f"Preset '{preset}' needs at least {dim_idx+1} dimensions, "
                f"got {len(dims)}."
            )
        normal = _face_normal(hkl, a_star, b_star, c_star)
        offset = float(dims[dim_idx])
        children.append(_halfspace_spec(normal, offset))

    shape_spec = {
        "type":     "intersection",
        "children": children,
    }

    lattice_meta = {
        "a_axis": a.tolist(),
        "b_axis": b.tolist(),
        "c_axis": c.tolist(),
    }

    return shape_spec, lattice_meta
