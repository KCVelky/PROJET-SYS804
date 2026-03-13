# models/plate.py

from __future__ import annotations

from dataclasses import dataclass, field

from utils.validators import ensure_positive, ensure_str_in
from models.material import Material
from models.black_hole import BlackHole


_ALLOWED_BC = (
    "simply_supported",
    "clamped",
    "free",
)


@dataclass
class Plate:
    """
    Plaque mince rectangulaire.
    Paramètres en SI :
    - length_x, length_y, thickness : m
    """

    length_x: float
    length_y: float
    thickness: float
    material: Material
    boundary_condition: str = "simply_supported"
    black_hole: BlackHole | None = None
    name: str = "Rectangular plate"

    def __post_init__(self) -> None:
        ensure_positive(self.length_x, "Longueur plaque")
        ensure_positive(self.length_y, "Largeur plaque")
        ensure_positive(self.thickness, "Épaisseur plaque")
        ensure_str_in(self.boundary_condition, "Condition limite", _ALLOWED_BC)

    @property
    def Lx(self) -> float:
        return self.length_x

    @property
    def Ly(self) -> float:
        return self.length_y

    @property
    def h0(self) -> float:
        return self.thickness

    @property
    def area(self) -> float:
        return self.length_x * self.length_y

    @property
    def surface_mass(self) -> float:
        return self.material.rho * self.thickness

    def thickness_at(self, x: float, y: float, use_black_hole: bool = True) -> float:
        if use_black_hole and self.black_hole is not None:
            return self.black_hole.thickness(x, y, self.thickness)
        return self.thickness

    def flexural_rigidity_at(self, x: float, y: float, use_black_hole: bool = True) -> float:
        """
        Rigidité de flexion isotrope Kirchhoff-Love :
        D = E h^3 / [12 (1 - nu^2)]
        """
        h = self.thickness_at(x, y, use_black_hole=use_black_hole)
        E = self.material.E
        nu = self.material.nu
        return E * h**3 / (12.0 * (1.0 - nu**2))

    def set_black_hole(self, black_hole: BlackHole | None) -> None:
        self.black_hole = black_hole

    def has_black_hole(self) -> bool:
        return self.black_hole is not None and self.black_hole.enabled

    def validate_black_hole_inside(self) -> None:
        if self.black_hole is None:
            return

        bh = self.black_hole
        if bh.xc - bh.radius < 0 or bh.xc + bh.radius > self.length_x:
            raise ValueError("Le trou noir sort de la plaque suivant x.")
        if bh.yc - bh.radius < 0 or bh.yc + bh.radius > self.length_y:
            raise ValueError("Le trou noir sort de la plaque suivant y.")
        if bh.residual_thickness >= self.thickness:
            raise ValueError(
                "L'épaisseur résiduelle du trou noir doit être inférieure à l'épaisseur nominale."
            )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "length_x": self.length_x,
            "length_y": self.length_y,
            "thickness": self.thickness,
            "boundary_condition": self.boundary_condition,
            "material": self.material.to_dict(),
            "black_hole": None if self.black_hole is None else self.black_hole.to_dict(),
        }