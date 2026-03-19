# solid3d/tet10_shape_functions.py

from __future__ import annotations

import numpy as np


class Tet10ShapeFunctions:
    """
    Fonctions de forme quadratiques Tetra10.

    Coordonnées naturelles :
    - r, s, t
    - l1 = 1 - r - s - t
    - l2 = r
    - l3 = s
    - l4 = t

    Ordre de noeuds adopté :
    1  : l1
    2  : l2
    3  : l3
    4  : l4
    5  : edge 1-2
    6  : edge 2-3
    7  : edge 3-1
    8  : edge 1-4
    9  : edge 2-4
    10 : edge 3-4
    """

    @staticmethod
    def shape_functions(xi: np.ndarray) -> np.ndarray:
        r, s, t = xi
        l1 = 1.0 - r - s - t
        l2 = r
        l3 = s
        l4 = t

        N = np.array(
            [
                l1 * (2.0 * l1 - 1.0),
                l2 * (2.0 * l2 - 1.0),
                l3 * (2.0 * l3 - 1.0),
                l4 * (2.0 * l4 - 1.0),
                4.0 * l1 * l2,
                4.0 * l2 * l3,
                4.0 * l3 * l1,
                4.0 * l1 * l4,
                4.0 * l2 * l4,
                4.0 * l3 * l4,
            ],
            dtype=float,
        )
        return N

    @staticmethod
    def shape_function_gradients_reference(xi: np.ndarray) -> np.ndarray:
        """
        Retourne dN/d(r,s,t) sous forme (10, 3).
        """
        r, s, t = xi
        l1 = 1.0 - r - s - t
        l2 = r
        l3 = s
        l4 = t

        grad_l = np.array(
            [
                [-1.0, -1.0, -1.0],  # grad l1
                [1.0, 0.0, 0.0],     # grad l2
                [0.0, 1.0, 0.0],     # grad l3
                [0.0, 0.0, 1.0],     # grad l4
            ],
            dtype=float,
        )

        l = [l1, l2, l3, l4]
        grads = []

        # noeuds sommets : Ni = li (2li - 1)
        for i in range(4):
            grads.append((4.0 * l[i] - 1.0) * grad_l[i])

        # noeuds milieux d'arêtes : Nij = 4 li lj
        edge_pairs = [
            (0, 1),  # 1-2
            (1, 2),  # 2-3
            (2, 0),  # 3-1
            (0, 3),  # 1-4
            (1, 3),  # 2-4
            (2, 3),  # 3-4
        ]

        for i, j in edge_pairs:
            grads.append(4.0 * (l[j] * grad_l[i] + l[i] * grad_l[j]))

        return np.array(grads, dtype=float)

    @staticmethod
    def quadrature_rule_4pt() -> tuple[np.ndarray, np.ndarray]:
        """
        Règle de quadrature 4 points sur le tétra de référence.

        Somme des poids = volume du tétra de référence = 1/6.
        """
        a = 0.1381966011250105
        b = 0.5854101966249685

        points = np.array(
            [
                [a, a, a],
                [b, a, a],
                [a, b, a],
                [a, a, b],
            ],
            dtype=float,
        )

        weights = np.full(4, 1.0 / 24.0, dtype=float)
        return points, weights