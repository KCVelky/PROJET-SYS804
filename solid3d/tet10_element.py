# solid3d/tet10_element.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solid3d.tet10_shape_functions import Tet10ShapeFunctions


@dataclass
class LinearElasticTet10Element:
    """
    Élément solide 3D tétraédrique quadratique Tetra10.
    DDL par noeud : [u, v, w]
    => 30 ddl par élément
    """

    young_modulus: float
    poisson_ratio: float
    density: float

    def __post_init__(self) -> None:
        if self.young_modulus <= 0.0:
            raise ValueError("Le module d'Young doit être > 0.")
        if not (0.0 <= self.poisson_ratio < 0.5):
            raise ValueError("Le coefficient de Poisson doit être dans [0, 0.5[.")
        if self.density <= 0.0:
            raise ValueError("La masse volumique doit être > 0.")

        self.D = self._elasticity_matrix()

    def _elasticity_matrix(self) -> np.ndarray:
        E = self.young_modulus
        nu = self.poisson_ratio

        lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        mu = E / (2.0 * (1.0 + nu))

        D = np.array(
            [
                [lam + 2.0 * mu, lam, lam, 0.0, 0.0, 0.0],
                [lam, lam + 2.0 * mu, lam, 0.0, 0.0, 0.0],
                [lam, lam, lam + 2.0 * mu, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, mu, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, mu, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, mu],
            ],
            dtype=float,
        )
        return D

    @staticmethod
    def build_B_matrix(dN_dx: np.ndarray) -> np.ndarray:
        """
        dN_dx : (10, 3) avec colonnes [dN/dx, dN/dy, dN/dz]
        """
        B = np.zeros((6, 30), dtype=float)

        for a in range(10):
            dNx, dNy, dNz = dN_dx[a]
            col = 3 * a

            B[0, col + 0] = dNx
            B[1, col + 1] = dNy
            B[2, col + 2] = dNz

            B[3, col + 0] = dNy
            B[3, col + 1] = dNx

            B[4, col + 1] = dNz
            B[4, col + 2] = dNy

            B[5, col + 0] = dNz
            B[5, col + 2] = dNx

        return B

    @staticmethod
    def build_N_matrix(N: np.ndarray) -> np.ndarray:
        """
        N : (10,)
        """
        Nmat = np.zeros((3, 30), dtype=float)
        for a in range(10):
            col = 3 * a
            Nmat[0, col + 0] = N[a]
            Nmat[1, col + 1] = N[a]
            Nmat[2, col + 2] = N[a]
        return Nmat

    def stiffness_and_mass(self, node_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        node_coords : (10, 3)
        """
        if node_coords.shape != (10, 3):
            raise ValueError("node_coords doit être de forme (10, 3).")

        Ke = np.zeros((30, 30), dtype=float)
        Me = np.zeros((30, 30), dtype=float)

        quad_points, quad_weights = Tet10ShapeFunctions.quadrature_rule_4pt()

        for xi, weight in zip(quad_points, quad_weights):
            N = Tet10ShapeFunctions.shape_functions(xi)
            dN_dxi = Tet10ShapeFunctions.shape_function_gradients_reference(xi)  # (10,3)

            # Jacobien de l'élément isoparamétrique
            J = node_coords.T @ dN_dxi  # (3,3)
            detJ = np.linalg.det(J)

            if detJ <= 1e-12:
                raise ValueError(f"Jacobien non positif ou trop petit rencontré : detJ = {detJ:.6e}")

            invJ = np.linalg.inv(J)
            dN_dx = dN_dxi @ invJ  # (10,3)

            B = self.build_B_matrix(dN_dx)
            Nmat = self.build_N_matrix(N)

            dV = detJ * weight

            Ke += (B.T @ self.D @ B) * dV
            Me += self.density * (Nmat.T @ Nmat) * dV

        return Ke, Me