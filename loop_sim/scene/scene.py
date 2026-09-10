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

def _build_shape(spec, device='cpu'):
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
            device=device,
        )
    if t == "surface_mesh":
        return SurfaceMesh(
            vertices=np.array(spec["vertices"], dtype=float),
            faces=np.array(spec["faces"],    dtype=int),
            device=device,
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
        children = [_build_shape(c, device) for c in spec["children"]]
        return Intersection(*children)
    if t == "union":
        children = [_build_shape(c, device) for c in spec["children"]]
        return Union(*children)
    if t == "difference":
        children = [_build_shape(c, device) for c in spec["children"]]
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
    # next_interface: closest t > t_min where material changes
    # ------------------------------------------------------------------

    def next_interface(self, origins, dirs, t_min=1e-6):
        """
        For each ray: find the closest interface at t > t_min.

        Parameters
        ----------
        origins : (N, 3)
        dirs    : (N, 3) unit vectors
        t_min   : float — ignore intersections closer than this

        Returns
        -------
        t          : (N,)    — distance to next interface (inf = no interface)
        normal     : (N, 3)  — outward surface normal at the interface
        mat_out_oi : (N,) int — object index of material after crossing
                                (-1 = background)
        """
        N = len(origins)
        n_obj = len(self.objects)
        best_t  = np.full(N, _INF)
        best_n  = np.zeros((N, 3))
        best_oi = np.full(N, -1, dtype=np.intp)
        all_te  = np.empty((N, n_obj))
        all_tx  = np.empty((N, n_obj))

        for oi, obj in enumerate(self.objects):
            te, tx, ne, nx = obj.shape.ray_intersect(origins, dirs)
            all_te[:, oi] = te
            all_tx[:, oi] = tx

            # Entry events at t > t_min
            valid_e = (te > t_min) & (te < tx) & (te < best_t)
            best_t[valid_e]  = te[valid_e]
            best_n[valid_e]  = ne[valid_e]
            best_oi[valid_e] = oi

            # Exit events at t > t_min
            valid_x = (tx > t_min) & (te < tx) & (tx < best_t)
            best_t[valid_x]  = tx[valid_x]
            best_n[valid_x]  = nx[valid_x]
            best_oi[valid_x] = oi

        # Determine material just past hit point via interval check.
        # t_probe = best_t + 1e-4 places the probe 100 nm past the interface.
        # 100 nm >> float32 ULP (~6 nm at t≈50 mm), so the probe reliably lands
        # on the correct side of the surface regardless of GPU rounding direction.
        # For an entry event (best_t = te_obj): te_obj < t_probe < tx_obj. ✓
        # For an exit  event (best_t = tx_obj): t_probe > tx_obj → not inside. ✓
        # No extra CUDA round-trips needed; all_te/all_tx are already computed.
        hit_mask = best_t < _INF
        mat_out_oi = np.full(N, -1, dtype=np.intp)
        if np.any(hit_mask):
            t_probe = best_t[hit_mask, None] + 1e-4   # (M, 1)
            te_m    = all_te[hit_mask]                 # (M, n_obj)
            tx_m    = all_tx[hit_mask]                 # (M, n_obj)
            inside  = (te_m < t_probe) & (t_probe < tx_m)   # (M, n_obj)
            has_any = inside.any(axis=1)               # (M,)
            mat_out_oi[hit_mask] = np.where(has_any, inside.argmax(axis=1), -1)

        # Recompute normals for GPU Tube hits in float64 to eliminate float32
        # edge speckle (~0.1% float32 normal error → ~0.03% with this fix).
        for oi, obj in enumerate(self.objects):
            if not (isinstance(obj.shape, Tube) and obj.shape._device != 'cpu'):
                continue
            tube_hit = hit_mask & (best_oi == oi)
            if not np.any(tube_hit):
                continue
            new_n = obj.shape.recompute_normals_f64(origins, dirs, best_t, tube_hit)
            best_n[tube_hit] = new_n[tube_hit]

        return best_t, best_n, mat_out_oi

    # ------------------------------------------------------------------
    # path traversal → ordered segments / per-material totals per ray
    # ------------------------------------------------------------------

    def _ray_segments(self, origins, dirs, t_max=200.0):
        """
        Walk all interfaces along each ray and return the ORDERED list of
        (Material, length_mm) segments per ray, front-to-back.

        Shared core for path_lengths() (which reduces the segments to a
        per-material dict) and path_segments() (which exposes the order).
        Order matters for X-ray Beer-Lambert attenuation, where a segment's
        incident flux depends on the cumulative attenuation of everything in
        front of it.

        Returns: list (one per ray) of list[(Material, float_mm)].
        """
        N = len(origins)
        segments = [[] for _ in range(N)]

        # Collect all (t, obj_idx, is_entry) events per ray
        all_events = []   # list of (t_array, obj_idx, is_entry_bool)
        for oi, obj in enumerate(self.objects):
            te, tx, _, _ = obj.shape.ray_intersect(origins, dirs)
            all_events.append((te, oi, True))
            all_events.append((tx, oi, False))

        n_obj = len(self.objects)
        for i in range(N):
            # Build sorted event list for ray i
            evs = []
            for t_arr, oi, is_entry in all_events:
                t = t_arr[i]
                if 0.0 < t < t_max:
                    evs.append((t, oi, is_entry))
            evs.sort(key=lambda x: x[0])

            # Walk events front-to-back
            active_objs = set()   # indices of objects currently containing the ray
            prev_t   = 0.0
            prev_mat = self.background
            for t_ev, oi, is_entry in evs:
                seg_len = t_ev - prev_t
                if seg_len > 0.0:
                    segments[i].append((prev_mat, seg_len))
                prev_t = t_ev
                if is_entry:
                    active_objs.add(oi)
                else:
                    active_objs.discard(oi)
                # highest-priority active object's material, else background
                prev_mat = self.background
                for oi2 in range(n_obj):   # priority order
                    if oi2 in active_objs:
                        prev_mat = self.objects[oi2].material
                        break

        return segments

    def path_lengths(self, origins, dirs, t_max=200.0):
        """
        Walk all interfaces along each ray and accumulate path lengths
        per material.

        Returns: list of dicts, one per ray: {Material: float_mm}
        """
        results = []
        for segs in self._ray_segments(origins, dirs, t_max):
            d = {}
            for mat, length in segs:
                d[mat] = d.get(mat, 0.0) + length
            results.append(d)
        return results

    def path_segments(self, origins, dirs, t_max=200.0):
        """
        Like path_lengths() but preserves front-to-back ORDER: returns, per
        ray, a list of (Material, length_mm) segments in traversal order.

        Needed for X-ray Beer-Lambert attenuation (see beam.py): each
        segment's incident flux is the incident beam attenuated by every
        segment ahead of it.
        """
        return self._ray_segments(origins, dirs, t_max)

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

def load(yaml_path, device='cpu'):
    """Load a scene YAML file and return a Scene.

    Parameters
    ----------
    yaml_path : str — path to the scene YAML file
    device    : str — 'cpu' (default) or 'cuda' for GPU-accelerated
                ray intersection in SurfaceMesh and Tube primitives.
    """
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
        shape = _build_shape(obj_spec["shape"], device=device)
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
