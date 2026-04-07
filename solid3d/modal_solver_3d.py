from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
    def __init__(self, fem_model: Solid3DFEMModel, verbose: bool = True) -> None:
        self.model = fem_model
        self.verbose = verbose

    def solve_basis(self, n_modes: int = 6) -> ModalBasis3D:
        if self.model.Kff is None or self.model.Mff is None:
            raise RuntimeError("Le modèle 3D doit être construit avant le calcul modal.")

        n_free = self.model.Kff.shape[0]
        if n_free <= 1:
            raise RuntimeError("Pas assez de ddl libres pour un calcul modal.")

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

        positive = eigvals > 1e-8
        eigvals = eigvals[positive]
        eigvecs = eigvecs[:, positive]

        order = np.argsort(eigvals)
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        if eigvals.size == 0:
            raise RuntimeError("Aucune valeur propre positive n'a été trouvée.")

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
            modal_masses.append(float(phi.T @ (self.model.Mff @ phi)))

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

    def save_basis(self, basis: ModalBasis3D, filepath: str | Path) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            frequencies_hz=basis.frequencies_hz,
            omegas_rad_s=basis.omegas_rad_s,
            eigenvalues=basis.eigenvalues,
            modes_free=basis.modes_free,
            modes_full=basis.modes_full,
            modal_masses=basis.modal_masses,
        )

        if self.verbose:
            print(f"Base modale 3D sauvegardée : {path}")

    def load_basis(self, filepath: str | Path) -> ModalBasis3D:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Base modale introuvable : {path}")

        data = np.load(path)

        basis = ModalBasis3D(
            frequencies_hz=data["frequencies_hz"],
            omegas_rad_s=data["omegas_rad_s"],
            eigenvalues=data["eigenvalues"],
            modes_free=data["modes_free"],
            modes_full=data["modes_full"],
            modal_masses=data["modal_masses"],
        )

        if self.verbose:
            print(f"Base modale 3D rechargée : {path}")

        return basis

    def truncate_basis(self, basis: ModalBasis3D, n_modes: int) -> ModalBasis3D:
        n_available = basis.modes_free.shape[1]
        n_keep = min(n_modes, n_available)

        return ModalBasis3D(
            frequencies_hz=basis.frequencies_hz[:n_keep],
            omegas_rad_s=basis.omegas_rad_s[:n_keep],
            eigenvalues=basis.eigenvalues[:n_keep],
            modes_free=basis.modes_free[:, :n_keep],
            modes_full=basis.modes_full[:, :n_keep],
            modal_masses=basis.modal_masses[:n_keep],
        )