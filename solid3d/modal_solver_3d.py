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
    modes_full: np.ndarray | None
    modal_masses: np.ndarray
    free_dofs: np.ndarray | None = None
    n_dofs_total: int | None = None

    def with_full_modes(self, model: Solid3DFEMModel) -> "ModalBasis3D":
        if self.modes_full is not None:
            return self
        if self.free_dofs is None:
            raise RuntimeError("Impossible de reconstruire les modes complets : free_dofs absent.")

        n_dofs_total = int(self.n_dofs_total) if self.n_dofs_total is not None else model.n_dofs
        modes_full = np.zeros((n_dofs_total, self.modes_free.shape[1]), dtype=self.modes_free.dtype)
        modes_full[self.free_dofs, :] = self.modes_free

        return ModalBasis3D(
            frequencies_hz=self.frequencies_hz,
            omegas_rad_s=self.omegas_rad_s,
            eigenvalues=self.eigenvalues,
            modes_free=self.modes_free,
            modes_full=modes_full,
            modal_masses=self.modal_masses,
            free_dofs=self.free_dofs,
            n_dofs_total=n_dofs_total,
        )


class ModalSolver3D:
    def __init__(self, fem_model: Solid3DFEMModel, verbose: bool = True) -> None:
        self.model = fem_model
        self.verbose = verbose

    def solve_basis(self, n_modes: int = 6, build_full_modes: bool = True) -> ModalBasis3D:
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

        modes_full = None
        if build_full_modes:
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
            free_dofs=None if self.model.free_dofs is None else self.model.free_dofs.copy(),
            n_dofs_total=self.model.n_dofs,
        )

    def solve(self, n_modes: int = 6) -> tuple[np.ndarray, np.ndarray | None]:
        basis = self.solve_basis(n_modes=n_modes)
        return basis.frequencies_hz, basis.modes_full

    def save_basis(
        self,
        basis: ModalBasis3D,
        filepath: str | Path,
        *,
        store_full_modes: bool = True,
        compressed: bool = False,
        dtype: np.dtype | str = np.float32,
    ) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload: dict[str, np.ndarray] = {
            "frequencies_hz": np.asarray(basis.frequencies_hz, dtype=float),
            "omegas_rad_s": np.asarray(basis.omegas_rad_s, dtype=float),
            "eigenvalues": np.asarray(basis.eigenvalues, dtype=float),
            "modes_free": np.asarray(basis.modes_free, dtype=dtype),
            "modal_masses": np.asarray(basis.modal_masses, dtype=float),
            "has_modes_full": np.array([int(store_full_modes and basis.modes_full is not None)], dtype=np.int8),
            "n_dofs_total": np.array([-1 if basis.n_dofs_total is None else int(basis.n_dofs_total)], dtype=np.int64),
        }
        if basis.free_dofs is not None:
            payload["free_dofs"] = np.asarray(basis.free_dofs, dtype=np.int64)
        if store_full_modes and basis.modes_full is not None:
            payload["modes_full"] = np.asarray(basis.modes_full, dtype=dtype)

        saver = np.savez_compressed if compressed else np.savez
        saver(path, **payload)

        if self.verbose:
            storage = "complet" if (store_full_modes and basis.modes_full is not None) else "compact"
            print(f"Base modale 3D sauvegardée ({storage}) : {path}")

    def load_basis(self, filepath: str | Path) -> ModalBasis3D:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Base modale introuvable : {path}")

        data = np.load(path)
        has_modes_full = bool(int(data["has_modes_full"][0])) if "has_modes_full" in data else ("modes_full" in data)
        n_dofs_total = int(data["n_dofs_total"][0]) if "n_dofs_total" in data else None
        if n_dofs_total is not None and n_dofs_total < 0:
            n_dofs_total = None

        basis = ModalBasis3D(
            frequencies_hz=np.asarray(data["frequencies_hz"], dtype=float),
            omegas_rad_s=np.asarray(data["omegas_rad_s"], dtype=float),
            eigenvalues=np.asarray(data["eigenvalues"], dtype=float),
            modes_free=np.asarray(data["modes_free"], dtype=float),
            modes_full=np.asarray(data["modes_full"], dtype=float) if has_modes_full and "modes_full" in data else None,
            modal_masses=np.asarray(data["modal_masses"], dtype=float),
            free_dofs=np.asarray(data["free_dofs"], dtype=np.int64) if "free_dofs" in data else None,
            n_dofs_total=n_dofs_total,
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
            modes_full=None if basis.modes_full is None else basis.modes_full[:, :n_keep],
            modal_masses=basis.modal_masses[:n_keep],
            free_dofs=basis.free_dofs,
            n_dofs_total=basis.n_dofs_total,
        )
