# solvers/modal_solver.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import eigsh

from fem.fem_model import FEMModel


@dataclass
class ModalBasis:
    """
    Base modale réduite.
    Les modes libres sont normalisés par rapport à la masse :
    phi_r^T M phi_r = 1
    """
    frequencies_hz: np.ndarray
    omegas_rad_s: np.ndarray
    eigenvalues: np.ndarray
    modes_free: np.ndarray
    modes_full: np.ndarray
    modal_masses: np.ndarray


class ModalSolver:
    """
    Solveur modal :
    Kff phi = lambda Mff phi
    avec lambda = omega^2
    """

    def __init__(self, fem_model: FEMModel) -> None:
        self.model = fem_model

    def solve_basis(self, n_modes: int = 6) -> ModalBasis:
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

        # normalisation masse : phi^T M phi = 1
        modal_masses = []
        for i in range(eigvecs.shape[1]):
            phi = eigvecs[:, i].copy()
            m_r = float(phi.T @ (self.model.Mff @ phi))

            if m_r <= 0.0:
                raise ValueError(f"Masse modale non positive pour le mode {i+1}.")

            phi /= np.sqrt(m_r)
            eigvecs[:, i] = phi

            m_r_check = float(phi.T @ (self.model.Mff @ phi))
            modal_masses.append(m_r_check)

        modal_masses = np.array(modal_masses, dtype=float)

        omegas = np.sqrt(eigvals)
        freqs_hz = omegas / (2.0 * np.pi)

        modes_full = np.column_stack([
            self.model.expand_reduced_vector(eigvecs[:, i]) for i in range(eigvecs.shape[1])
        ])

        return ModalBasis(
            frequencies_hz=freqs_hz,
            omegas_rad_s=omegas,
            eigenvalues=eigvals,
            modes_free=eigvecs,
            modes_full=modes_full,
            modal_masses=modal_masses,
        )

    def solve(self, n_modes: int = 6) -> tuple[np.ndarray, np.ndarray]:
        basis = self.solve_basis(n_modes=n_modes)
        return basis.frequencies_hz, basis.modes_full