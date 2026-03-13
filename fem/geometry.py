# fem/geometry.py

from __future__ import annotations

import numpy as np

from models.plate import Plate


class PlateGeometry:
    """
    Géométrie continue de la plaque.
    Fournit le champ d'épaisseur et le champ de rigidité sur une grille 2D.
    """

    def __init__(self, plate: Plate) -> None:
        self.plate = plate

    def is_inside(self, x: float, y: float) -> bool:
        return (0.0 <= x <= self.plate.Lx) and (0.0 <= y <= self.plate.Ly)

    def thickness_at(self, x: float, y: float, use_black_hole: bool = True) -> float:
        return self.plate.thickness_at(x, y, use_black_hole=use_black_hole)

    def flexural_rigidity_at(self, x: float, y: float, use_black_hole: bool = True) -> float:
        return self.plate.flexural_rigidity_at(x, y, use_black_hole=use_black_hole)

    def make_grid(self, nx: int = 201, ny: int = 161) -> tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0.0, self.plate.Lx, nx)
        y = np.linspace(0.0, self.plate.Ly, ny)
        return np.meshgrid(x, y)

    def thickness_field(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        use_black_hole: bool = True
    ) -> np.ndarray:
        """
        Calcule le champ d'épaisseur h(x, y) sur une grille 2D.
        Version vectorisée.
        """
        h0 = self.plate.h0

        if (not use_black_hole) or (self.plate.black_hole is None) or (not self.plate.black_hole.enabled):
            return np.full_like(X, h0, dtype=float)

        bh = self.plate.black_hole

        r = np.sqrt((X - bh.xc) ** 2 + (Y - bh.yc) ** 2)
        H = np.full_like(X, h0, dtype=float)

        mask_core = r <= bh.truncation_radius
        H[mask_core] = bh.residual_thickness

        mask_profile = (r > bh.truncation_radius) & (r <= bh.radius)
        xi = (r[mask_profile] - bh.truncation_radius) / (bh.radius - bh.truncation_radius)
        H[mask_profile] = bh.residual_thickness + (h0 - bh.residual_thickness) * (xi ** bh.exponent)

        return H

    def flexural_rigidity_field(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        use_black_hole: bool = True
    ) -> np.ndarray:
        """
        Champ de rigidité de flexion D(x, y).
        """
        h = self.thickness_field(X, Y, use_black_hole=use_black_hole)
        E = self.plate.material.E
        nu = self.plate.material.nu
        return E * h**3 / (12.0 * (1.0 - nu**2))

    def black_hole_boundary(self, n_points: int = 300) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Renvoie le contour du rayon extérieur du trou noir.
        """
        bh = self.plate.black_hole
        if bh is None:
            return None

        theta = np.linspace(0.0, 2.0 * np.pi, n_points)
        x = bh.xc + bh.radius * np.cos(theta)
        y = bh.yc + bh.radius * np.sin(theta)
        return x, y

    def truncation_boundary(self, n_points: int = 300) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Renvoie le contour du rayon de troncature.
        """
        bh = self.plate.black_hole
        if bh is None:
            return None

        theta = np.linspace(0.0, 2.0 * np.pi, n_points)
        x = bh.xc + bh.truncation_radius * np.cos(theta)
        y = bh.yc + bh.truncation_radius * np.sin(theta)
        return x, y