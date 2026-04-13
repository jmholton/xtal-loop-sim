"""
MiTeGen Micromount presets.

MiTeGen Micromounts are flat Kapton (polyimide) sheets cut into specific
shapes and slightly curved for dimensional stability.  They have no nylon
fiber, no twisted stem, and no fiber diffraction contribution.

Each model number maps to a geometry specification (2D outline waypoints,
thickness, and sag).  The outline is a closed loop in the sheet plane,
digitised from MiTeGen product drawings.

The builder emits a `thin_shell` scene object for the renderer.

Usage
-----
    from crystal_harvester.mitegen_mounts import build_mitegen_scene
    import yaml

    scene_dict = build_mitegen_scene(
        model          = 'M2-L18SP-200',
        crystal_preset = 'hexagonal',
        crystal_dims_mm = [0.08, 0.04],
        lattice_abc     = {'a_axis':[10.5,0,0], 'b_axis':[0,10.5,0], 'c_axis':[0,0,27.3]},
    )
    with open('mitegen_scene.yaml', 'w') as f:
        yaml.dump(scene_dict, f)
"""

from typing import List, Optional
import numpy as np

from .crystal_shapes import make_crystal
from .pin_geometry   import make_pin


# ---------------------------------------------------------------------------
# Preset data type
# ---------------------------------------------------------------------------

class MiTeGenPreset:
    """
    Geometry spec for one MiTeGen Micromount model.

    outline_2d : list of [x, z] points (mm) in the sheet plane.
                 The sheet normal is Y; fast axis is X; slow axis is Z.
                 The aperture opening faces +Z.
    """
    def __init__(self, aperture_um, outline_2d,
                 thickness_mm=0.007, sag_mm=0.005, description=""):
        self.aperture_um  = aperture_um
        self.outline_2d   = outline_2d
        self.thickness_mm = thickness_mm
        self.sag_mm       = sag_mm
        self.description  = description


# ---------------------------------------------------------------------------
# Helper: build a teardrop 2D outline
# ---------------------------------------------------------------------------

def _teardrop_outline(aperture_mm, width_mm=None, n=32):
    """
    Closed 2D outline for a teardrop-shaped aperture.

    The aperture opening is at z = +aperture_mm/2; the pointed tail at z < 0.
    X spans ±width_mm/2.

    Returns list of [x, z] points (mm).
    """
    if width_mm is None:
        width_mm = aperture_mm * 1.3

    # Upper semicircle (aperture end)
    theta_top = np.linspace(-np.pi / 2, np.pi / 2, n // 2)
    x_top = (width_mm / 2) * np.cos(theta_top)
    z_top = aperture_mm / 2 + (aperture_mm / 2) * np.sin(theta_top)

    # Lower taper to a point
    x_bot = np.linspace(width_mm / 2, 0.0, n // 4)
    z_bot = np.linspace(-aperture_mm * 0.1, -aperture_mm * 0.4, n // 4)

    x_bot2 = np.linspace(0.0, -width_mm / 2, n // 4)
    z_bot2 = np.linspace(-aperture_mm * 0.4, -aperture_mm * 0.1, n // 4)

    x = np.concatenate([x_top, x_bot[1:], x_bot2[1:]])
    z = np.concatenate([z_top, z_bot[1:], z_bot2[1:]])

    return [[float(xi), float(zi)] for xi, zi in zip(x, z)]


def _round_outline(aperture_mm, n=48):
    """
    Closed 2D outline for a round/circular aperture.

    Returns list of [x, z] points (mm).
    """
    theta = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    r = aperture_mm / 2.0
    x = r * np.cos(theta)
    z = r * np.sin(theta) + r   # centred at z = r so aperture spans 0..2r
    return [[float(xi), float(zi)] for xi, zi in zip(x, z)]


# ---------------------------------------------------------------------------
# Model number lookup table
# ---------------------------------------------------------------------------
# Aperture sizes are per MiTeGen catalogue.
# Outline shapes are approximated from product images; exact geometry would
# require digitising MiTeGen CAD drawings.

_MODELS = {}

for _ap in [10, 20, 50]:
    _k = f"M2-L18SP-{_ap}"
    _ap_mm = _ap / 1000.0
    _MODELS[_k] = MiTeGenPreset(
        aperture_um   = _ap,
        outline_2d    = _round_outline(_ap_mm),
        thickness_mm  = 0.005,
        sag_mm        = 0.002,
        description   = f"MiTeGen Micromount round {_ap} µm",
    )

for _ap in [100, 200, 400, 600, 800, 1000]:
    _k = f"M2-L18SP-{_ap}"
    _ap_mm = _ap / 1000.0
    _t = 0.007 if _ap <= 200 else 0.010
    _MODELS[_k] = MiTeGenPreset(
        aperture_um   = _ap,
        outline_2d    = _teardrop_outline(_ap_mm),
        thickness_mm  = _t,
        sag_mm        = 0.005,
        description   = f"MiTeGen Micromount teardrop {_ap} µm",
    )

# Aliases
_MODELS["M2-L18SP-10"]  = _MODELS["M2-L18SP-10"] if "M2-L18SP-10" in _MODELS else _MODELS["M2-L18SP-20"]


# ---------------------------------------------------------------------------
# Default scene parameters (shared with hampton_loops)
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
    "pixel_size":   0.005,
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
    "crystal": {"n": 1.52, "mu_optical": 0.02, "mu_xray": 2.1,   "color": [0.7, 0.9, 1.0]},
    "solvent": {"n": 1.34, "mu_optical": 0.00, "mu_xray": 0.3,   "color": [0.2, 0.4, 0.8]},
    "kapton":  {"n": 1.70, "mu_optical": 0.08, "mu_xray": 0.005, "color": [0.9, 0.7, 0.2]},
    "metal":   {"n": 2.50, "mu_optical": 500., "mu_xray": 100.,  "color": [0.7, 0.7, 0.8]},
    "air":     {"n": 1.00, "mu_optical": 0.00, "mu_xray": 0.0,   "color": [1.0, 1.0, 1.0]},
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_models():
    """Return sorted list of available MiTeGen model numbers."""
    return sorted(_MODELS.keys())


def build_mitegen_scene(
    model               : str,
    crystal_preset      : Optional[str]  = "hexagonal",
    crystal_dims_mm     : Optional[list] = None,
    lattice_abc         : Optional[dict] = None,
    pin_diameter_mm     : float = 0.5,
    pin_length_mm       : float = 6.0,
    pin_bevel_deg       : float = 45.0,
    geometry            : dict  = None,
    camera              : dict  = None,
    beam                : dict  = None,
    materials           : dict  = None,
) -> dict:
    """
    Build a complete scene dict for a MiTeGen Micromount assembly.

    Parameters
    ----------
    model           : str  — MiTeGen model number, e.g. 'M2-L18SP-200'
    crystal_preset  : str  — crystal habit preset (or None to omit crystal)
    crystal_dims_mm : list — half-widths for the chosen preset
    lattice_abc     : dict — XDS real-space unit cell vectors in Angstroms
    pin_diameter_mm : float
    pin_length_mm   : float
    pin_bevel_deg   : float

    Returns
    -------
    dict — YAML-serialisable scene dict
    """
    if model not in _MODELS:
        available = list_models()
        raise ValueError(f"Unknown MiTeGen model {model!r}. "
                         f"Available: {available}")

    preset = _MODELS[model]
    ap_mm  = preset.aperture_um / 1000.0

    geo  = {**DEFAULT_GEOMETRY, **(geometry  or {})}
    cam  = {**DEFAULT_CAMERA,   **(camera    or {})}
    bm   = {**DEFAULT_BEAM,     **(beam      or {})}
    mats = {**DEFAULT_MATERIALS, **(materials or {})}

    # --- Micromount sheet (ThinShell) ---
    mount_shape = {
        "type":      "thin_shell",
        "outline":   preset.outline_2d,
        "thickness": round(preset.thickness_mm, 6),
        "sag":       round(preset.sag_mm, 6),
        "normal":    [0.0, 1.0, 0.0],   # sheet normal = Y
        "fast":      [1.0, 0.0, 0.0],   # sheet X = lab X
        "slow":      [0.0, 0.0, 1.0],   # sheet Z = lab Z
    }

    objects = []

    # --- Crystal ---
    if crystal_preset is not None:
        if crystal_dims_mm is None:
            r = ap_mm / 3.0
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
        objects.append({
            "name":     "crystal",
            "material": "crystal",
            "lattice":  xtal_lattice,
            "shape":    xtal_shape,
        })

    # Micromount (after crystal so crystal takes priority)
    objects.append({
        "name":     "micromount",
        "material": "kapton",
        "shape":    mount_shape,
    })

    # Pin
    pin_shape = make_pin(pin_diameter_mm, pin_length_mm, pin_bevel_deg)
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
