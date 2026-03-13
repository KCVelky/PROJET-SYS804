# models/black_hole.py

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

from utils.validators import ensure_positive, ensure_non_negative


@dataclass
class BlackHole:
    """
    Trou noir vibratoire axisymétrique tronqué.

    Paramètres en SI :
    - xc, yc, radius, truncation_radius, residual_thickness : m
    - exponent : sans dimension
    """

    xc: float
    yc: float
    radius: float
    truncation_radius: float
    residual_thickness: float
    exponent: float = 2.0
    enabled: bool = True

    def __post_init__(self) -> None:
        ensure_non_negative(self.xc, "xc")
        ensure_non_negative(self.yc, "yc")
        ensure_positive(self.radius, "Rayon extérieur du trou noir")
        ensure_non_negative(self.truncation_radius, "Rayon de troncature")
        ensure_positive(self.residual_thickness, "Épaisseur résiduelle")
        ensure_positive(self.exponent, "Exposant du profil")

        if self.truncation_radius >= self.radius:
            raise ValueError(
                "Le rayon de troncature doit être strictement inférieur au rayon extérieur."
            )

    def radial_distance(self, x: float, y: float) -> float:
        return sqrt((x - self.xc) ** 2 + (y - self.yc) ** 2)

    def thickness(self, x: float, y: float, h0: float) -> float:
        """
        Loi d'épaisseur h(r) :
        - ht si r <= a
        - ht + (h0 - ht) * ((r-a)/(R-a))^m si a < r <= R
        - h0 sinon
        """
        if not self.enabled:
            return h0

        r = self.radial_distance(x, y)

        if r <= self.truncation_radius:
            return self.residual_thickness

        if r <= self.radius:
            xi = (r - self.truncation_radius) / (self.radius - self.truncation_radius)
            return self.residual_thickness + (h0 - self.residual_thickness) * (xi ** self.exponent)

        return h0

    def contains(self, x: float, y: float) -> bool:
        return self.radial_distance(x, y) <= self.radius

    def to_dict(self) -> dict:
        return {
            "xc": self.xc,
            "yc": self.yc,
            "radius": self.radius,
            "truncation_radius": self.truncation_radius,
            "residual_thickness": self.residual_thickness,
            "exponent": self.exponent,
            "enabled": self.enabled,
        }