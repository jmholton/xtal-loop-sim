"""
Hampton Research CryoLoop presets.

Hampton CryoLoops are nylon fiber loops with a twisted-pair stem glued to a
metal pin.  Each model is parameterised by loop diameter and fiber diameter;
the stem geometry is computed from the fiber mechanics.

Calling build_hampton_scene() returns a complete YAML scene dict that can be
serialised with yaml.dump() and passed to loop_sim.scene.scene.load().

Usage
-----
    from crystal_harvester.hampton_loops import build_hampton_scene
    import yaml

    scene_dict = build_hampton_scene(
        loop_diameter_um = 300,
        fiber_diameter_um = 20,
        solvent_volume_mm3 = 0.002,
        contact_angle_deg  = 30.0,
        crystal_preset     = 'hexagonal',
        crystal_dims_mm    = [0.06, 0.03],
        lattice_abc        = {'a_axis':[10.5,0,0], 'b_axis':[0,10.5,0], 'c_axis':[0,0,27.3]},
    )
    with open('scene.yaml', 'w') as f:
        yaml.dump(scene_dict, f, default_flow_style=None)
"""

from typing import Optional
import numpy as np
import yaml

from .nylon_mechanics import helix_path, elastica_loop
from .droplet          import droplet_in_loop
from .crystal_shapes   import make_crystal
from .pin_geometry     import make_pin


def _rotation_from_to(a, b):
    """
    3×3 rotation matrix that rotates unit vector *a* to unit vector *b*.
    Uses the Rodrigues formula; handles parallel and anti-parallel cases.
    """
    a = np.asarray(a, dtype=float); a = a / np.linalg.norm(a)
    b = np.asarray(b, dtype=float); b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-10:
        if c > 0:
            return np.eye(3)
        # Anti-parallel: rotate 180° about any perpendicular axis
        perp = np.array([1., 0., 0.]) if abs(a[0]) < 0.9 else np.array([0., 1., 0.])
        perp -= np.dot(perp, a) * a;  perp /= np.linalg.norm(perp)
        return 2.0 * np.outer(perp, perp) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + (vx @ vx) * (1.0 - c) / (s * s)


# ---------------------------------------------------------------------------
# Preset catalogue
# ---------------------------------------------------------------------------

class HamptonPreset:
    """Physical parameters for a Hampton CryoLoop model."""
    def __init__(self, loop_diameter_um, fiber_diameter_um=20.0,
                 loop_shape="teardrop", stem_length_mm=0.5,
                 stem_pitch_ratio=5.0, pin_diameter_mm=0.5,
                 pin_length_mm=6.0, pin_bevel_deg=0.0,
                 youngs_modulus_gpa=2.5):
        self.loop_diameter_um   = loop_diameter_um
        self.fiber_diameter_um  = fiber_diameter_um
        self.loop_shape         = loop_shape
        self.stem_length_mm     = stem_length_mm
        self.stem_pitch_ratio   = stem_pitch_ratio
        self.pin_diameter_mm    = pin_diameter_mm
        self.pin_length_mm      = pin_length_mm
        self.pin_bevel_deg      = pin_bevel_deg
        self.youngs_modulus_gpa = youngs_modulus_gpa


# Standard Hampton CryoLoop sizes (loop diameter in µm)
HAMPTON_PRESETS = {
    100:  HamptonPreset(loop_diameter_um=100,  fiber_diameter_um=10,  pin_diameter_mm=0.7, stem_length_mm=0.7, stem_pitch_ratio=20),
    200:  HamptonPreset(loop_diameter_um=200,  fiber_diameter_um=15,  pin_diameter_mm=0.7, stem_length_mm=0.7, stem_pitch_ratio=20),
    300:  HamptonPreset(loop_diameter_um=300,  fiber_diameter_um=20,  pin_diameter_mm=0.7, stem_length_mm=0.7, stem_pitch_ratio=20),
    400:  HamptonPreset(loop_diameter_um=400,  fiber_diameter_um=20,  pin_diameter_mm=0.7, stem_length_mm=0.7, stem_pitch_ratio=20),
    500:  HamptonPreset(loop_diameter_um=500,  fiber_diameter_um=25,  pin_diameter_mm=0.7, stem_length_mm=0.7, stem_pitch_ratio=20),
    600:  HamptonPreset(loop_diameter_um=600,  fiber_diameter_um=25,  pin_diameter_mm=0.7, stem_length_mm=0.7, stem_pitch_ratio=20),
    800:  HamptonPreset(loop_diameter_um=800,  fiber_diameter_um=30,  pin_diameter_mm=0.7, stem_length_mm=0.7, stem_pitch_ratio=20),
    1000: HamptonPreset(loop_diameter_um=1000, fiber_diameter_um=30,  pin_diameter_mm=0.7, stem_length_mm=0.7, stem_pitch_ratio=20),
}


# ---------------------------------------------------------------------------
# Default scene parameters
# ---------------------------------------------------------------------------

DEFAULT_GEOMETRY = {
    "beam_axis":    [0, 0, 1],
    "optical_axis": [0, 0, -1],
    "camera_fast":  [1, 0, 0],
    "camera_slow":  [0, 1, 0],
    "rotx_axis":    [1, 0, 0],
    "roty_axis":    [0, 1, 0],
    "rotz_axis":    [0, 0, 1],
}

DEFAULT_CAMERA = {
    "width":        640,
    "height":       480,
    "pixel_size":   0.0074,   # 7.4 µm, matches real Hampton loop camera
    "na_objective": 0.10,
    "na_condenser": 0.07,
}

DEFAULT_BEAM = {
    "spacing": 0.001,
    "profile": "gaussian",
    "fwhm_x":  0.05,
    "fwhm_y":  0.03,
}

DEFAULT_MATERIALS = {
    # color is an ABSORPTION spectrum in the renderer: mu_per_ch = mu_optical
    # + 30*(1-color) per mm (microscope.py).  A coloured material is an
    # absorbing material; keep water-like solvents near-white.  The crystal's
    # mu_optical = 4.09/mm (color white) reproduces the luma of a neutral
    # reference frame through refraction (n = 1.52) and the NA 0.10 gate; see
    # docs/DECISIONS.md 2026-08-10, the scene changes that produced the NA
    # evidence.
    "crystal": {"n": 1.52, "mu_optical": 4.09, "mu_xray": 2.1,  "color": [1.0, 1.0, 1.0]},
    "solvent": {"n": 1.34, "mu_optical": 0.00, "mu_xray": 0.3,  "color": [0.97, 0.98, 1.0]},
    "nylon":   {"n": 1.53, "mu_optical": 0.10, "mu_xray": 0.1,  "color": [0.9, 0.8, 0.6]},
    "metal":   {"n": 2.50, "mu_optical": 500., "mu_xray": 100., "color": [0.7, 0.7, 0.8]},
    "air":     {"n": 1.00, "mu_optical": 0.00, "mu_xray": 0.0,  "color": [1.0, 1.0, 1.0]},
}


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_hampton_scene(
    loop_diameter_um    : int   = 300,
    fiber_diameter_um   : float = None,   # None → from preset
    loop_shape          : str   = None,   # None → from preset
    stem_length_mm      : float = None,   # None → from preset
    stem_pitch_ratio    : float = None,   # None → from preset
    pin_diameter_mm     : float = None,   # None → from preset
    pin_length_mm       : float = None,
    pin_bevel_deg       : float = None,
    youngs_modulus_gpa  : float = None,
    solvent_volume_mm3  : float = 0.00893,
    contact_angle_deg   : float = 30.0,
    # Droplet tessellation. The defaults give ~10 um facets on a 300 um loop,
    # which is 3x COARSER than the objective's 3.35 um Rayleigh limit at
    # NA 0.10 -- invisible at the camera's own 7.4 um pixel, but a visible
    # staircase on the drop's edge once a supersampled library lets you zoom
    # in. Raise both together (they are the two axes of one revolve) when the
    # library's template pixel is finer than the facet size.
    drop_mesh_nz        : int   = 30,
    drop_mesh_nphi      : int   = 48,
    crystal_preset      : Optional[str]  = "hexagonal",
    crystal_dims_mm     : Optional[list] = None,
    lattice_abc         : Optional[dict] = None,
    loop_axis           : list  = None,   # direction FROM pin TOWARD loop; default [-1,0,0]
    pin_axis            : list  = None,   # direction pin points (base→tip); default = loop_axis
    stem_axis           : list  = None,   # DEPRECATED: old convention (loop→pin = +X); kept for compat
    geometry            : dict = None,
    camera              : dict = None,
    beam                : dict = None,
    materials           : dict = None,
) -> dict:
    """
    Build a complete scene dict for a Hampton CryoLoop assembly.

    Returns a dict that can be passed directly to yaml.dump().

    contact_angle_deg is accepted for API compatibility but IGNORED: the
    droplet rim is pinned at the loop fiber, so the contact angle is
    determined by volume and rim radius, not prescribable.
    """
    # --- Resolve preset ---
    preset = HAMPTON_PRESETS.get(loop_diameter_um)
    if preset is None:
        preset = HamptonPreset(loop_diameter_um=loop_diameter_um)

    # `x if x is not None else preset.x`, NOT `x or preset.x`.  Every one of
    # these is a number whose legitimate values include 0, and `or` silently
    # discards a caller's 0 in favour of the preset.  That made `--pin-bevel 0`
    # (a flat cut, per its --help) a lie: it returned a 45-degree chisel, and
    # the same bug was latent on the five neighbours.  No behaviour changes
    # today, because passing 0 never worked and so nobody does: this only
    # makes the flags mean what they say.
    def _or_preset(value, name):
        return getattr(preset, name) if value is None else value

    fd_mm   = _or_preset(fiber_diameter_um, "fiber_diameter_um") / 1000.0
    ld_mm   = preset.loop_diameter_um / 1000.0
    shape   = _or_preset(loop_shape,         "loop_shape")
    stem_l  = _or_preset(stem_length_mm,     "stem_length_mm")
    pitch_r = _or_preset(stem_pitch_ratio,   "stem_pitch_ratio")
    pin_d   = _or_preset(pin_diameter_mm,    "pin_diameter_mm")
    pin_l   = _or_preset(pin_length_mm,      "pin_length_mm")
    pin_bv  = _or_preset(pin_bevel_deg,      "pin_bevel_deg")
    E_gpa   = _or_preset(youngs_modulus_gpa, "youngs_modulus_gpa")

    # --- Geometry ---
    geo = {**DEFAULT_GEOMETRY, **(geometry or {})}
    cam = {**DEFAULT_CAMERA,   **(camera   or {})}
    bm  = {**DEFAULT_BEAM,     **(beam     or {})}
    mats = {**DEFAULT_MATERIALS, **(materials or {})}

    # --- Axis directions ---
    # Convention: loop_axis points FROM the pin base TOWARD the loop (leftward in a typical setup).
    # stem (internal) goes in the opposite direction: from loop attachment toward pin.
    if loop_axis is not None:
        lax = np.asarray(loop_axis, dtype=float)
    elif stem_axis is not None:
        # Old convention: stem_axis pointed from loop toward pin (+X).  Reverse it.
        lax = -np.asarray(stem_axis, dtype=float)
    else:
        lax = np.array([-1.0, 0.0, 0.0])
    lax = lax / np.linalg.norm(lax)

    # Pin axis: direction the pin points (base→tip).  Defaults to loop_axis.
    if pin_axis is not None:
        pax = np.asarray(pin_axis, dtype=float)
        pax = pax / np.linalg.norm(pax)
    else:
        pax = lax

    # Internal stem direction: from loop attachment toward pin (opposite of loop_axis)
    sax = -lax

    # --- Loop fiber waypoints ---
    # elastica_loop generates the loop in the canonical frame: loop extends in −X,
    # attachment at origin, loop in XY plane.  Rotate to align with lax.
    loop_pts_canon = elastica_loop(fd_mm, ld_mm, E_gpa, shape, n_points=30)
    R_rot = _rotation_from_to(np.array([-1., 0., 0.]), lax)
    loop_pts = (R_rot @ loop_pts_canon.T).T

    # --- Stem (twisted pair): two helical fibers ---
    # The stem is glued to the pin's scored break face.  make_pin puts that
    # face bevel_offset past tip_pos (it crosses the stem axis exactly there),
    # so the fibers must run PAST the tip and into the metal to emerge from
    # the face at every angle: ending them at tip_pos leaves them floating
    # 0.3 mm in front of it.  The overrun is swallowed by the opaque pin.
    pin_bevel_offset = 0.3   # passed to make_pin below; keep the two in step
    stem_run = stem_l + pin_bevel_offset + 2 * fd_mm
    R_helix = fd_mm / 2      # fibers touch: 2R = fd_mm (center-to-center = diameter)
    pitch   = pitch_r * fd_mm
    stem1   = helix_path(R_helix, pitch, stem_run, n_points=40, phase_offset=0.0,    axis=sax)
    stem2   = helix_path(R_helix, pitch, stem_run, n_points=40, phase_offset=np.pi,  axis=sax)

    # Translate stem to attach at loop origin
    stem_offset = loop_pts[0]   # attachment point of loop (≈ origin after rotation)
    stem1 = stem1 + stem_offset
    stem2 = stem2 + stem_offset

    # --- Pin: tip at stem far end; body extends in −pax direction from tip ---
    tip_pos = (stem_offset + sax * stem_l).tolist()
    pin_shape = make_pin(pin_d, pin_l, pin_bv, bevel_offset_mm=pin_bevel_offset,
                         tip_pos=tip_pos, axis=(-pax).tolist())

    # --- Solvent droplet: biconvex lens pinned in the loop aperture ---
    # Built in the canonical frame (loop in the z=0 plane) so the rim ray-cast
    # is 2-D, then rotated by the same R_rot as the loop so the droplet stays
    # coplanar with it for any loop_axis.  The rim follows the actual loop
    # outline via a dense elastica polygon (the 30 tube waypoints undershoot
    # the curved fiber between samples).  An unpinnable volume raises: there
    # is deliberately no fallback shape (contact_angle_deg is ignored: with
    # the rim pinned at the loop, contact angle is an output of volume + rim
    # radius, not an input).
    dense_canon = elastica_loop(fd_mm, ld_mm, E_gpa, shape, n_points=240)
    sol_verts_c, sol_faces, drop_info = droplet_in_loop(
        loop_pts_canon, fd_mm, solvent_volume_mm3,
        n_z=int(drop_mesh_nz), n_phi=int(drop_mesh_nphi),
        dense_poly_xy=dense_canon[:, :2])
    sol_verts = (R_rot @ sol_verts_c.T).T
    # Crystal goes at the droplet's VOLUME centroid, not the waypoint
    # centroid: on an asymmetric (teardrop) aperture the liquid body's centre
    # sits toward the wide side, and that is where a crystal would rest.
    aperture_centre = R_rot @ np.asarray(drop_info["volume_centroid"], dtype=float)
    solvent_shape = {
        "type":     "surface_mesh",
        "vertices": [[round(float(x), 5) for x in row] for row in sol_verts],
        "faces":    sol_faces.tolist(),
    }

    # --- Crystal ---
    objects = []

    if crystal_preset is not None:
        if crystal_dims_mm is None:
            # Default: ~1/3 of loop diameter
            r = ld_mm / 3.0
            crystal_dims_mm = [r, r / 2.0]
        if lattice_abc is None:
            lattice_abc = {
                "a_axis": [10.5, 0.0,  0.0],
                "b_axis": [ 0.0, 10.5, 0.0],
                "c_axis": [ 0.0,  0.0, 27.3],
            }
        xtal_shape, xtal_lattice = make_crystal(
            crystal_preset, crystal_dims_mm, lattice_abc
        )
        # Place the crystal at the aperture centre, inside the droplet: a
        # half-space translates by offset += normal . centre.
        for hs in xtal_shape["children"]:
            n_hs = np.asarray(hs["normal"], dtype=float)
            hs["offset"] = round(float(hs["offset"] + n_hs @ aperture_centre), 8)
        objects.append({
            "name":     "crystal",
            "material": "crystal",
            "lattice":  xtal_lattice,
            "shape":    xtal_shape,
        })

    # Solvent (before nylon so crystal overrides in priority)
    objects.append({
        "name":     "solvent",
        "material": "solvent",
        "shape":    solvent_shape,
    })

    # Loop fiber
    objects.append({
        "name":     "loop_fiber",
        "material": "nylon",
        "shape": {
            "type":     "tube",
            "diameter": round(fd_mm, 6),
            "path":     loop_pts.tolist(),
        },
    })

    # Stem fibers
    objects.append({
        "name":     "stem_fiber_1",
        "material": "nylon",
        "shape": {
            "type":     "tube",
            "diameter": round(fd_mm, 6),
            "path":     stem1.tolist(),
        },
    })
    objects.append({
        "name":     "stem_fiber_2",
        "material": "nylon",
        "shape": {
            "type":     "tube",
            "diameter": round(fd_mm, 6),
            "path":     stem2.tolist(),
        },
    })

    # Pin
    objects.append({
        "name":     "pin",
        "material": "metal",
        "shape":    pin_shape,
    })

    return {
        "geometry":  geo,
        "camera":    cam,
        "beam":      bm,
        "materials": mats,
        "objects":   objects,
    }
