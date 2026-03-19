# solid3d/modal_solver_3d.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import eigsh

from solid3d.fem_model_3d import Solid3DFEMModel


@dataclass
class ModalBasis3D:
    frequencies_hz: np.ndarray
    omegas_rad_s: np.ndarray
    eigenvalues: np.ndarray
    modes_free: np.ndarray
    modes_full: np.ndarray
    modal_masses: np.ndarray


class ModalSolver3D:
    """
    Solveur modal 3D :
    Kff phi = lambda Mff phi
    """

    def __init__(self, fem_model: Solid3DFEMModel, verbose: bool = True) -> None:
        self.model = fem_model
        self.verbose = verbose

    def solve_basis(self, n_modes: int = 6) -> ModalBasis3D:
        if self.model.Kff is None or self.model.Mff is None:
            raise RuntimeError("Le modèle 3D doit être construit avant le calcul modal.")

        n_free = self.model.Kff.shape[0]
        if n_free <= 1:
            raise RuntimeError("Pas assez de ddl libres pour un calcul modal.")

        # Pour un cas free, on peut récupérer des modes rigides proches de 0 ;
        # on demande un peu plus de modes pour avoir assez de modes positifs.
        extra = 6 if self.model.plate.boundary_condition == "free" else 0
        k = min(max(1, n_modes + extra), n_free - 2)

        if self.verbose:
            print("=== Calcul modal 3D ===")
            print("Nombre de modes demandés :", n_modes)
            print("Nombre de vecteurs ARPACK :", k)

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

        # filtre des valeurs propres positives
        positive = eigvals > 1e-8
        eigvals = eigvals[positive]
        eigvecs = eigvecs[:, positive]

        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        if eigvals.size == 0:
            raise RuntimeError("Aucune valeur propre positive n'a été trouvée.")

        # on tronque au nombre demandé
        eigvals = eigvals[:n_modes]
        eigvecs = eigvecs[:, :n_modes]

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

        modes_full = np.column_stack(
            [self.model.expand_reduced_vector(eigvecs[:, i]) for i in range(eigvecs.shape[1])]
        )

        return ModalBasis3D(
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