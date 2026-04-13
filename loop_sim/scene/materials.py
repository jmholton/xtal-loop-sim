"""
Material optical and X-ray properties.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Material:
    name: str
    n: float                        # refractive index (optical)
    mu_optical: float               # absorption coefficient (mm⁻¹), visible light
    mu_xray: float                  # absorption coefficient (mm⁻¹), X-rays
    color: Tuple[float, float, float]  # display RGB in [0, 1]

    def __repr__(self):
        return f"Material({self.name!r}, n={self.n}, mu_opt={self.mu_optical})"


# Default background material — used for any point not inside any scene object.
AIR = Material(
    name="air",
    n=1.000,
    mu_optical=0.0,
    mu_xray=0.0,
    color=(1.0, 1.0, 1.0),
)

# Typical values for common cryo-crystallography materials.
WATER = Material(
    name="water",
    n=1.333,
    mu_optical=0.0,
    mu_xray=0.03,
    color=(0.85, 0.92, 1.0),
)

NYLON = Material(
    name="nylon",
    n=1.530,
    mu_optical=0.10,
    mu_xray=0.10,
    color=(0.9, 0.85, 0.7),
)

KAPTON = Material(
    name="kapton",
    n=1.700,
    mu_optical=0.08,
    mu_xray=0.005,   # Kapton is an X-ray window material — very low absorption
    color=(0.9, 0.7, 0.2),
)

METAL = Material(
    name="metal",
    n=2.50,
    mu_optical=500.0,  # effectively opaque
    mu_xray=100.0,
    color=(0.7, 0.72, 0.78),
)
