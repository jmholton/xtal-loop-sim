"""
Scene: loads a YAML scene file and provides ray-casting services.

Object priority
---------------
Objects are listed in the YAML in priority order: earlier entries override
later ones at any point in space.  Typically: crystal > solvent > nylon > air.

next_interface(origins, dirs, t_min)
    For each ray: find the closest interface at t > t_min where the material
    changes.  Returns (t, normal, material_from, material_to).
    Used by the Snell's law ray-tracer in microscope.py.

path_lengths(origins, dirs)
    For each ray: walk all interfaces and accumulate {material: length}.
    Used by the X-ray beam reporter in beam.py.
"""
import numpy as np
import yaml

from .materials import Material, AIR
from .primitives import (Sphere, Cylinder, InfiniteCylinder,
                         HalfSpace, Ellipsoid, Box, Capsule)
from .tube         import Tube
from .surface_mesh import SurfaceMesh
from .thin_shell   import ThinShell
from .csg          import Intersection, Union, Difference

_INF = np.inf


# ---------------------------------------------------------------------------
# YAML → shape builders
# ---------------------------------------------------------------------------

def _build_shape(spec):
    """Recursively build a shape/CSG object from a YAML spec dict."""
    t = spec["type"]

    if t == "sphere":
        return Sphere(
            centre=spec.get("centre", [0, 0, 0]),
            radius=spec["radius"],
        )
    if t == "half_space":
        return HalfSpace(
            normal=spec["normal"],
            offset=spec["offset"],
        )
    if t == "cylinder":
        return Cylinder(
            centre=spec.get("centre", [0, 0, 0]),
            axis=spec.get("axis", [0, 1, 0]),
            radius=spec["radius"],
            height=spec["height"],
        )
    if t == "infinite_cylinder":
        return InfiniteCylinder(
            centre=spec.get("centre", [0, 0, 0]),
            axis=spec.get("axis", [0, 1, 0]),
            radius=spec["radius"],
        )
    if t == "ellipsoid":
        return Ellipsoid(
            centre=spec.get("centre", [0, 0, 0]),
            radii=spec["radii"],
        )
    if t == "box":
        return Box(lo=spec["lo"], hi=spec["hi"])
    if t == "capsule":
        return Capsule(p0=spec["p0"], p1=spec["p1"], radius=spec["radius"])
    if t == "tube":
        return Tube(
            waypoints=spec["path"],
            diameter=spec["diameter"],
            n_samples=spec.get("n_samples", 50),
        )
    if t == "surface_mesh":
        return SurfaceMesh(
            vertices=np.array(spec["vertices"], dtype=float),
            faces=np.array(spec["faces"],    dtype=int),
        )
    if t == "thin_shell":
        return ThinShell(
            outline_2d=spec["outline"],
            thickness=spec.get("thickness", 0.007),
            sag=spec.get("sag", 0.0),
            normal=spec.get("normal", [0, 1, 0]),
            fast=spec.get("fast",   [1, 0, 0]),
            slow=spec.get("slow",   [0, 0, 1]),
            n_outline=spec.get("n_outline", 60),
        )
    if t == "intersection":
        children = [_build_shape(c) for c in spec["children"]]
        return Intersection(*children)
    if t == "union":
        children = [_build_shape(c) for c in spec["children"]]
        return Union(*children)
    if t == "difference":
        children = [_build_shape(c) for c in spec["children"]]
        return Difference(children[0], children[1])

    raise ValueError(f"Unknown shape type: {t!r}")


# ---------------------------------------------------------------------------
# SceneObject
# ---------------------------------------------------------------------------

class SceneObject:
    def __init__(self, name, shape, material, is_fiber=False):
        self.name      = name
        self.shape     = shape
        self.material  = material
        self.is_fiber  = is_fiber   # True for Tube objects → export tangents


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------

class Scene:
    """
    Loaded scene.  `objects` is ordered by priority (index 0 = highest).
    """

    def __init__(self, objects, geometry, camera_cfg, beam_cfg, background=AIR):
        self.objects    = objects      # list[SceneObject], highest priority first
        self.geometry   = geometry     # dict of axis vectors
        self.camera_cfg = camera_cfg   # dict
        self.beam_cfg   = beam_cfg     # dict
        self.background = background

    # ------------------------------------------------------------------
    # Material at a point (for tracking current medium)
    # ------------------------------------------------------------------

    def material_at(self, point):
        """Return the highest-priority material containing `point`."""
        p = np.asarray(point, dtype=float)
        for obj in self.objects:
            # Use a very short ray in an arbitrary direction to test containment
            o = p[None, :]
            d = np.array([[0., 0., 1.]])
            te, tx, _, _ = obj.shape.ray_intersect(o, d)
            if te[0] < 0.0 < tx[0]:   # origin is inside
                return obj.material
        return self.background

    # ------------------------------------------------------------------
    # next_interface: closest t > t_min where material changes
    # ------------------------------------------------------------------

    def next_interface(self, origins, dirs, current_materials, t_min=1e-6):
        """
        For each ray: find the closest interface at t > t_min.

        Parameters
        ----------
        origins           : (N, 3)
        dirs              : (N, 3) unit vectors
        current_materials : list/array of Material, length N
        t_min             : float — ignore intersections closer than this

        Returns
        -------
        t      : (N,)   — distance to next interface (inf = no interface)
        normal : (N, 3) — outward surface normal at the interface
        mat_in : list[Material], length N — material before crossing
        mat_out: list[Material], length N — material after crossing
        """
        N = len(origins)
        best_t  = np.full(N, _INF)
        best_n  = np.zeros((N, 3))
        best_entering = np.ones(N, dtype=bool)
        best_obj_idx  = np.full(N, -1, dtype=int)

        for oi, obj in enumerate(self.objects):
            te, tx, ne, nx = obj.shape.ray_intersect(origins, dirs)

            # Entry events at t > t_min
            valid_e = (te > t_min) & (te < tx) & (te < best_t)
            best_t[valid_e]   = te[valid_e]
            best_n[valid_e]   = ne[valid_e]
            best_obj_idx[valid_e] = oi
            best_entering[valid_e] = True

            # Exit events at t > t_min
            valid_x = (tx > t_min) & (te < tx) & (tx < best_t)
            best_t[valid_x]   = tx[valid_x]
            best_n[valid_x]   = nx[valid_x]
            best_obj_idx[valid_x] = oi
            best_entering[valid_x] = False

        # Build material lists — batch probe all hit points at once
        mat_in  = list(current_materials)
        mat_out = [self.background] * N

        hit_mask = best_t < _INF
        if np.any(hit_mask):
            hit_idx   = np.where(hit_mask)[0]
            probe_pts = (origins[hit_mask]
                         + best_t[hit_mask, None] * dirs[hit_mask]
                         + dirs[hit_mask] * 1e-7)
            probed = self._material_at_points_batch(probe_pts)
            for j, i in enumerate(hit_idx):
                mat_out[i] = probed[j]

        return best_t, best_n, mat_in, mat_out

    def _material_at_point(self, point):
        """Scalar version: material at a single 3-D point."""
        o = point[None, :]
        d = np.zeros((1, 3)); d[0, 2] = 1.0
        for obj in self.objects:
            te, tx, _, _ = obj.shape.ray_intersect(o, d)
            if te[0] < 0.0 < tx[0]:
                return obj.material
        return self.background

    def _material_at_points_batch(self, points):
        """
        Vectorized material probe: returns list[Material] of length len(points).

        For each point, returns the highest-priority material whose shape
        contains that point (te < 0 < tx for a probe ray along +Z).
        """
        M = len(points)
        probe_d = np.zeros((M, 3))
        probe_d[:, 2] = 1.0          # probe along +Z for all points
        result_idx = np.full(M, -1, dtype=int)

        for oi, obj in enumerate(self.objects):
            te, tx, _, _ = obj.shape.ray_intersect(points, probe_d)
            inside = (te < 0.0) & (tx > 0.0) & (result_idx == -1)
            result_idx[inside] = oi
            if np.all(result_idx >= 0):
                break

        return [self.objects[i].material if i >= 0 else self.background
                for i in result_idx]

    # ------------------------------------------------------------------
    # path_lengths: full traversal → {material: total_length} per ray
    # ------------------------------------------------------------------

    def path_lengths(self, origins, dirs, t_max=200.0):
        """
        Walk all interfaces along each ray and accumulate path lengths
        per material.

        Returns: list of dicts, one per ray: {Material: float_mm}
        """
        N = len(origins)
        results = [{} for _ in range(N)]

        # Collect all (t, obj_idx, is_entry) events per ray
        all_events = []   # list of (t_array, obj_idx, is_entry_bool)
        for oi, obj in enumerate(self.objects):
            te, tx, _, _ = obj.shape.ray_intersect(origins, dirs)
            all_events.append((te, oi, True))
            all_events.append((tx, oi, False))

        for i in range(N):
            # Build sorted event list for ray i
            evs = []
            for t_arr, oi, is_entry in all_events:
                t = t_arr[i]
                if 0.0 < t < t_max:
                    evs.append((t, oi, is_entry))
            evs.sort(key=lambda x: x[0])

            # Walk events
            active_objs = set()   # indices of objects currently containing the ray
            prev_t = 0.0

            def current_mat():
                for oi2 in range(len(self.objects)):  # priority order
                    if oi2 in active_objs:
                        return self.objects[oi2].material
                return self.background

            prev_mat = self.background
            for t_ev, oi, is_entry in evs:
                seg_len = t_ev - prev_t
                if seg_len > 0.0:
                    mat = prev_mat
                    results[i][mat] = results[i].get(mat, 0.0) + seg_len
                prev_t = t_ev
                if is_entry:
                    active_objs.add(oi)
                else:
                    active_objs.discard(oi)
                prev_mat = current_mat()

        return results

    # ------------------------------------------------------------------
    # Fiber-axis data for beam reporter
    # ------------------------------------------------------------------

    def fiber_objects(self):
        """Return list of (SceneObject, Tube) for fiber-type objects."""
        out = []
        for obj in self.objects:
            if obj.is_fiber and isinstance(obj.shape, Tube):
                out.append((obj, obj.shape))
        return out


# ---------------------------------------------------------------------------
# YAML loader
# ---------------------------------------------------------------------------

def load(yaml_path):
    """Load a scene YAML file and return a Scene."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    # Materials
    mat_data = data.get("materials", {})
    mat_lookup = {}
    for name, props in mat_data.items():
        mat_lookup[name] = Material(
            name=name,
            n=props.get("n", 1.0),
            mu_optical=props.get("mu_optical", 0.0),
            mu_xray=props.get("mu_xray", 0.0),
            color=tuple(props.get("color", [1.0, 1.0, 1.0])),
        )

    # Objects (priority: order in YAML = highest first)
    objects = []
    for obj_spec in data.get("objects", []):
        mat_name = obj_spec["material"]
        mat = mat_lookup.get(mat_name, AIR)
        shape = _build_shape(obj_spec["shape"])
        is_fiber = obj_spec["shape"]["type"] == "tube"
        # Preserve lattice metadata as attribute if present
        sobj = SceneObject(
            name=obj_spec.get("name", ""),
            shape=shape,
            material=mat,
            is_fiber=is_fiber,
        )
        if "lattice" in obj_spec:
            sobj.lattice = {
                k: np.array(v, dtype=float)
                for k, v in obj_spec["lattice"].items()
            }
        objects.append(sobj)

    geometry   = data.get("geometry", {})
    camera_cfg = data.get("camera", {})
    beam_cfg   = data.get("beam", {})

    return Scene(objects, geometry, camera_cfg, beam_cfg)
