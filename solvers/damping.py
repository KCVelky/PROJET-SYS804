# solvers/damping.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class RayleighDamping:
    """
    Amortissement de Rayleigh :
    C = alpha * M + beta * K
    """

    alpha: float = 0.0
    beta: float = 0.0

    @classmethod
    def from_modal_damping_ratio(
        cls,
        zeta: float,
        freq1_hz: float,
        freq2_hz: float,
    ) -> "RayleighDamping":
        """
        Détermine alpha et beta en imposant le même taux d'amortissement zeta
        à deux fréquences de référence.
        """
        if zeta < 0.0:
            raise ValueError("Le taux d'amortissement doit être positif ou nul.")
        if freq1_hz <= 0.0 or freq2_hz <= 0.0:
            raise ValueError("Les fréquences de référence doivent être strictement positives.")
        if np.isclose(freq1_hz, freq2_hz):
            raise ValueError("Les deux fréquences de référence doivent être différentes.")

        w1 = 2.0 * np.pi * freq1_hz
        w2 = 2.0 * np.pi * freq2_hz

        A = np.array([
            [1.0 / (2.0 * w1), w1 / 2.0],
            [1.0 / (2.0 * w2), w2 / 2.0],
        ], dtype=float)

        b = np.array([zeta, zeta], dtype=float)

        alpha, beta = np.linalg.solve(A, b)
        return cls(alpha=float(alpha), beta=float(beta))

    def matrix(self, M: csr_matrix, K: csr_matrix) -> csr_matrix:
        return self.alpha * M + self.beta * K