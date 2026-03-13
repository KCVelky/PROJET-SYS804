# solvers/modal_solver.py

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh

from fem.fem_model import FEMModel


class ModalSolver:
    """
    Solveur modal :
    Kff phi = lambda Mff phi
    avec lambda = omega^2
    """

    def __init__(self, fem_model: FEMModel) -> None:
        self.model = fem_model

    def solve(self, n_modes: int = 6) -> tuple[np.ndarray, np.ndarray]:
        if self.model.Kff is None or self.model.Mff is None:
            raise RuntimeError("Le FEMModel doit être construit avant le calcul modal.")

        n_free = self.model.Kff.shape[0]
        if n_free <= 1:
            raise RuntimeError("Pas assez de ddl libres pour un calcul modal.")

        k = min(n_modes, max(1, n_free - 1))

        try:
            eigvals, eigvecs = eigsh(
                self.model.Kff,
                k=k,
                M=self.model.Mff,
                sigma=0.0,
                which="LM",
            )
        except Exception:
            eigvals, eigvecs = eigsh(
                self.model.Kff,
                k=k,
                M=self.model.Mff,
                which="SM",
            )

        eigvals = np.real(eigvals)
        eigvecs = np.real(eigvecs)

        positive = eigvals > 1e-9
        eigvals = eigvals[positive]
        eigvecs = eigvecs[:, positive]

        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        omegas = np.sqrt(eigvals)
        freqs_hz = omegas / (2.0 * np.pi)

        full_modes = np.column_stack([
            self.model.expand_reduced_vector(eigvecs[:, i]) for i in range(eigvecs.shape[1])
        ])

        return freqs_hz, full_modes