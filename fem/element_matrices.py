# fem/element_matrices.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ElementData:
    area: float
    dNdx: np.ndarray   # shape (3,)
    dNdy: np.ndarray   # shape (3,)
    thickness: float


class MindlinTri3Element:
    """
    Élément triangulaire linéaire de plaque Mindlin-Reissner.
    DDL par noeud : [w, theta_x, theta_y]
    => 9 ddl par élément.

    Version de démarrage :
    - rigidité de flexion + cisaillement transverse
    - masse cohérente simplifiée
    """

    def __init__(self, young_modulus: float, poisson_ratio: float, density: float, shear_correction: float = 5.0/6.0) -> None:
        self.E = young_modulus
        self.nu = poisson_ratio
        self.rho = density
        self.kappa = shear_correction

    @staticmethod
    def compute_geometry(node_coords: np.ndarray, thickness: float) -> ElementData:
        """
        node_coords : array (3,2)
        """
        x1, y1 = node_coords[0]
        x2, y2 = node_coords[1]
        x3, y3 = node_coords[2]

        detJ2 = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
        area = 0.5 * detJ2

        if area <= 0:
            raise ValueError("Élément triangulaire dégénéré ou orienté négativement.")

        b = np.array([y2 - y3, y3 - y1, y1 - y2], dtype=float)
        c = np.array([x3 - x2, x1 - x3, x2 - x1], dtype=float)

        dNdx = b / (2.0 * area)
        dNdy = c / (2.0 * area)

        return ElementData(
            area=area,
            dNdx=dNdx,
            dNdy=dNdy,
            thickness=thickness,
        )

    def bending_matrix(self, h: float) -> np.ndarray:
        """
        Matrice matériau de flexion.
        """
        coeff = self.E * h**3 / (12.0 * (1.0 - self.nu**2))
        return coeff * np.array([
            [1.0, self.nu, 0.0],
            [self.nu, 1.0, 0.0],
            [0.0, 0.0, (1.0 - self.nu) / 2.0],
        ], dtype=float)

    def shear_matrix(self, h: float) -> np.ndarray:
        """
        Matrice matériau de cisaillement transverse.
        """
        G = self.E / (2.0 * (1.0 + self.nu))
        coeff = self.kappa * G * h
        return coeff * np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ], dtype=float)

    @staticmethod
    def build_B_bending(dNdx: np.ndarray, dNdy: np.ndarray) -> np.ndarray:
        """
        Courbures :
        kappa_x  = d(theta_x)/dx
        kappa_y  = d(theta_y)/dy
        kappa_xy = d(theta_x)/dy + d(theta_y)/dx
        """
        B = np.zeros((3, 9), dtype=float)

        for i in range(3):
            col = 3 * i
            B[0, col + 1] = dNdx[i]
            B[1, col + 2] = dNdy[i]
            B[2, col + 1] = dNdy[i]
            B[2, col + 2] = dNdx[i]

        return B

    @staticmethod
    def build_B_shear(dNdx: np.ndarray, dNdy: np.ndarray) -> np.ndarray:
        """
        Cisaillements :
        gamma_xz = theta_x + dw/dx
        gamma_yz = theta_y + dw/dy

        Pour T3 linéaire, au centroïde N_i = 1/3.
        """
        B = np.zeros((2, 9), dtype=float)
        N = 1.0 / 3.0

        for i in range(3):
            col = 3 * i
            B[0, col + 0] = dNdx[i]
            B[0, col + 1] = N

            B[1, col + 0] = dNdy[i]
            B[1, col + 2] = N

        return B

    @staticmethod
    def scalar_consistent_mass(area: float) -> np.ndarray:
        """
        Masse cohérente scalaire T3 :
        A/12 * [[2,1,1],[1,2,1],[1,1,2]]
        """
        return (area / 12.0) * np.array([
            [2.0, 1.0, 1.0],
            [1.0, 2.0, 1.0],
            [1.0, 1.0, 2.0],
        ], dtype=float)

    def stiffness_matrix(self, node_coords: np.ndarray, thickness: float) -> np.ndarray:
        geom = self.compute_geometry(node_coords, thickness)

        B_b = self.build_B_bending(geom.dNdx, geom.dNdy)
        B_s = self.build_B_shear(geom.dNdx, geom.dNdy)

        D_b = self.bending_matrix(thickness)
        D_s = self.shear_matrix(thickness)

        K_b = geom.area * (B_b.T @ D_b @ B_b)
        K_s = geom.area * (B_s.T @ D_s @ B_s)

        return K_b + K_s

    def mass_matrix(self, node_coords: np.ndarray, thickness: float) -> np.ndarray:
        geom = self.compute_geometry(node_coords, thickness)
        area = geom.area

        C = self.scalar_consistent_mass(area)

        M = np.zeros((9, 9), dtype=float)

        # masse transverse
        Mw = self.rho * thickness * C

        # inertie de rotation
        Jr = self.rho * thickness**3 / 12.0
        Mr = Jr * C

        for i in range(3):
            for j in range(3):
                M[3*i + 0, 3*j + 0] = Mw[i, j]
                M[3*i + 1, 3*j + 1] = Mr[i, j]
                M[3*i + 2, 3*j + 2] = Mr[i, j]

        return M