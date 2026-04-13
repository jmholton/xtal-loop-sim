"""
ThinShell primitive: a thin curved sheet defined by

    - A closed 2-D Neville outline in the local (u, w) plane
    - A thickness t  (extruded in the local v / normal direction)
    - An optional spherical sag s  (dome height at centre, for slight curvature)

At construction time the outline is tessellated into a triangulated surface
mesh (front face, back face, and edge band) and stored as a SurfaceMesh.

Coordinate convention
---------------------
The sheet lies roughly in the plane spanned by the `fast` and `slow` vectors.
The `normal` vector is the extrusion direction.

MiTeGen Micromounts are the primary use case.
"""
import numpy as np
from .tube import neville_sample
from .surface_mesh import SurfaceMesh


def _revolve_outline(outline_2d, normal, fast, slow, thickness, sag, n_edge=60):
    """
    Build a triangulated mesh from a 2-D closed outline.

    outline_2d : (m, 2) array — closed 2-D waypoints in (fast, slow) coords (mm)
    normal     : (3,)  unit vector — extrusion / thickness direction
    fast       : (3,)  unit vector — first in-plane axis
    slow       : (3,)  unit vector — second in-plane axis
    thickness  : float (mm)
    sag        : float — dome height at centre (positive = dome toward +normal)
    n_edge     : int   — number of outline sample points
    """
    outline_2d = np.asarray(outline_2d, dtype=float)
    # Resample outline via Neville
    pts_2d = neville_sample(outline_2d, n_edge)  # (n_edge, 2)
    # Centre of mass (used for sag calculation)
    centre_2d = pts_2d.mean(axis=0)
    # Radii from centre (for sag offset)
    r_sq = np.sum((pts_2d - centre_2d) ** 2, axis=1)
    r_max_sq = r_sq.max()

    # Compute sag offset per point (spherical dome approximation)
    # sag_offset(r) = sag * (1 - r² / r_max²)  — zero at boundary, max at centre
    sag_offset = sag * (1.0 - r_sq / (r_max_sq + 1e-30))  # (n_edge,)

    def to_3d(pts_2d_local, v_offset):
        """Convert 2-D outline coords to 3-D lab points."""
        return (pts_2d_local[:, 0:1] * fast[None, :]
                + pts_2d_local[:, 1:2] * slow[None, :]
                + v_offset[:, None]    * normal[None, :])   # (n_edge, 3)

    # Front face (+ normal side)
    v_front = thickness / 2.0 + sag_offset
    pts_front = to_3d(pts_2d, v_front)

    # Back face (- normal side)
    v_back  = -thickness / 2.0 + sag_offset
    pts_back  = to_3d(pts_2d, v_back)

    # Build vertex array: front ring, then back ring
    # V = 2 * n_edge vertices
    vertices = np.vstack([pts_front, pts_back])   # (2*n_edge, 3)
    n = n_edge

    faces = []
    # --- Front face (fan triangulation from first point) ---
    for i in range(1, n - 2):
        faces.append([0, i, i + 1])

    # --- Back face (fan, reversed winding for outward normals) ---
    for i in range(1, n - 2):
        faces.append([n, n + i + 1, n + i])

    # --- Edge band: quads split into triangles connecting front and back rings ---
    for i in range(n):
        j = (i + 1) % n
        fi = i
        fj = j
        bi = n + i
        bj = n + j
        faces.append([fi, bi, fj])
        faces.append([bi, bj, fj])

    faces = np.array(faces, dtype=int)
    return vertices, faces


class ThinShell:
    """
    Thin curved Kapton (or similar) sheet.

    Parameters
    ----------
    outline_2d : array-like, shape (m, 2)
        Closed 2-D waypoints in (u, w) = (fast, slow) coordinates (mm).
        The first and last point should be the same (closed loop).
    thickness  : float — sheet thickness (mm)
    sag        : float — dome height at centre (mm); 0 = flat
    normal     : array-like, shape (3,) — extrusion direction in sample frame
    fast       : array-like, shape (3,) — first in-plane axis
    slow       : array-like, shape (3,) — second in-plane axis
    n_outline  : int — Neville resample count for the outline
    """

    def __init__(self, outline_2d, thickness=0.007, sag=0.0,
                 normal=(0., 1., 0.), fast=(1., 0., 0.), slow=(0., 0., 1.),
                 n_outline=60):
        self.outline_2d = np.asarray(outline_2d, dtype=float)
        self.thickness  = float(thickness)
        self.sag        = float(sag)
        self.normal     = np.asarray(normal, dtype=float)
        self.normal    /= np.linalg.norm(self.normal)
        self.fast       = np.asarray(fast,   dtype=float)
        self.fast      /= np.linalg.norm(self.fast)
        self.slow       = np.asarray(slow,   dtype=float)
        self.slow      /= np.linalg.norm(self.slow)

        verts, faces = _revolve_outline(
            self.outline_2d, self.normal, self.fast, self.slow,
            self.thickness, self.sag, n_edge=n_outline
        )
        self._mesh = SurfaceMesh(verts, faces)

    def ray_intersect(self, origins, dirs):
        return self._mesh.ray_intersect(origins, dirs)
