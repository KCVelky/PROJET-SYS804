# solid3d/geometry_abh_3d.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.plate import Plate


@dataclass
class PlateABH3DGeometry:
    """
    Géométrie 3D réelle de la plaque.

    Convention géométrique :
    - face inférieure plane : z = 0
    - face supérieure usinée : z = h(x, y)

    Donc le volume de la plaque est :
        0 <= z <= h(x, y)

    Cette géométrie correspond à un usinage du trou noir vibratoire
    par le dessus uniquement.
    """

    plate: Plate
    use_black_hole: bool = True

    @property
    def Lx(self) -> float:
        return self.plate.Lx

    @property
    def Ly(self) -> float:
        return self.plate.Ly

    @property
    def h0(self) -> float:
        return self.plate.h0

    def bottom_z(self, x: float, y: float) -> float:
        return 0.0

    def top_z(self, x: float, y: float) -> float:
        return self.plate.thickness_at(x, y, use_black_hole=self.use_black_hole)

    def thickness_at(self, x: float, y: float) -> float:
        return self.top_z(x, y) - self.bottom_z(x, y)

    def make_top_grid(self, nu: int = 21, nv: int = 17) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Grille régulière de points de contrôle / échantillonnage sur la face supérieure.
        """
        x = np.linspace(0.0, self.Lx, nu)
        y = np.linspace(0.0, self.Ly, nv)
        X, Y = np.meshgrid(x, y)

        Z = np.zeros_like(X, dtype=float)
        for j in range(nv):
            for i in range(nu):
                Z[j, i] = self.top_z(float(X[j, i]), float(Y[j, i]))

        return X, Y, Z

    def black_hole_center(self) -> tuple[float, float, float] | None:
        bh = self.plate.black_hole
        if bh is None:
            return None
        zc = 0.5 * self.h0
        return bh.xc, bh.yc, zc