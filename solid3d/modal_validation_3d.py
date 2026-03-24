from __future__ import annotations

import numpy as np

from solid3d.fem_model_3d import Solid3DFEMModel
from solid3d.modal_solver_3d import ModalBasis3D, ModalSolver3D
from solid3d.results_3d import ModalValidationPerMode3D, ModalValidationResult3D


class ModalValidation3D:
    """
    Validation numérique légère du solveur modal 3D.

    Le but n'est pas de refaire le solveur, mais de vérifier automatiquement :
    - l'orthogonalité modale selon M
    - la cohérence Phi^T K Phi ~ diag(lambda)
    - les résidus modaux
    - l'absence de noeuds orphelins dans le maillage stocké
    """

    def __init__(self, model: Solid3DFEMModel, verbose: bool = True) -> None:
        self.model = model
        self.verbose = verbose

    def validate(
        self,
        n_modes: int = 6,
        modal_basis: ModalBasis3D | None = None,
    ) -> ModalValidationResult3D:
        if self.model.Kff is None or self.model.Mff is None or self.model.mesh is None:
            raise RuntimeError("Le modèle 3D doit être construit avant validation.")

        if modal_basis is None:
            solver = ModalSolver3D(self.model, verbose=self.verbose)
            modal_basis = solver.solve_basis(n_modes=n_modes)

        Phi = modal_basis.modes_free
        Mff = self.model.Mff
        Kff = self.model.Kff

        mass_ortho = np.asarray(Phi.T @ (Mff @ Phi), dtype=float)
        stiff_ortho = np.asarray(Phi.T @ (Kff @ Phi), dtype=float)

        per_mode: list[ModalValidationPerMode3D] = []
        for i in range(Phi.shape[1]):
            phi = Phi[:, i]
            lam = float(modal_basis.eigenvalues[i])
            freq_hz = float(modal_basis.frequencies_hz[i])
            modal_mass = float(phi.T @ (Mff @ phi))

            residual = Kff @ phi - lam * (Mff @ phi)
            residual_norm = float(np.linalg.norm(residual))
            lhs_norm = float(np.linalg.norm(Kff @ phi))
            residual_rel = residual_norm / max(1.0, lhs_norm)

            k_cons = float(phi.T @ (Kff @ phi) - lam)
            k_cons_rel = abs(k_cons) / max(1.0, abs(lam))

            per_mode.append(
                ModalValidationPerMode3D(
                    mode_number=i + 1,
                    frequency_hz=freq_hz,
                    eigenvalue=lam,
                    modal_mass=modal_mass,
                    mass_normalization_error=abs(modal_mass - 1.0),
                    stiffness_consistency_error=k_cons,
                    stiffness_consistency_error_rel=k_cons_rel,
                    residual_norm=residual_norm,
                    residual_rel=residual_rel,
                )
            )

        used_nodes = np.unique(self.model.mesh.cells.ravel())
        n_used_nodes = int(len(used_nodes))
        n_unused_nodes = int(self.model.mesh.n_points - n_used_nodes)

        result = ModalValidationResult3D(
            n_nodes=int(self.model.mesh.n_points),
            n_elements=int(self.model.mesh.n_cells),
            n_dofs=int(self.model.n_dofs),
            n_free_dofs=int(len(self.model.free_dofs) if self.model.free_dofs is not None else 0),
            n_constrained_dofs=int(len(self.model.constrained_dofs) if self.model.constrained_dofs is not None else 0),
            n_used_nodes=n_used_nodes,
            n_unused_nodes=n_unused_nodes,
            frequencies_hz=np.asarray(modal_basis.frequencies_hz, dtype=float),
            mass_orthogonality_matrix=mass_ortho,
            stiffness_orthogonality_matrix=stiff_ortho,
            per_mode=per_mode,
        )

        if self.verbose:
            self.print_summary(result)

        return result

    @staticmethod
    def is_valid(
        result: ModalValidationResult3D,
        mass_tol: float = 1e-6,
        stiffness_rel_tol: float = 1e-6,
        residual_tol: float = 1e-6,
        allow_unused_nodes: bool = False,
    ) -> bool:
        if result.max_mass_normalization_error > mass_tol:
            return False
        if result.max_mass_offdiag > mass_tol:
            return False
        if result.max_stiffness_consistency_error_rel > stiffness_rel_tol:
            return False
        if result.max_stiffness_offdiag_rel > stiffness_rel_tol:
            return False
        if result.max_residual_rel > residual_tol:
            return False
        if not allow_unused_nodes and result.n_unused_nodes != 0:
            return False
        return True

    @staticmethod
    def print_summary(result: ModalValidationResult3D) -> None:
        print("\n=== Validation modale 3D ===")
        print(f"Noeuds                     : {result.n_nodes}")
        print(f"Éléments                   : {result.n_elements}")
        print(f"DDL totaux                 : {result.n_dofs}")
        print(f"DDL libres                 : {result.n_free_dofs}")
        print(f"DDL bloqués                : {result.n_constrained_dofs}")
        print(f"Noeuds utilisés            : {result.n_used_nodes}")
        print(f"Noeuds inutilisés          : {result.n_unused_nodes}")
        print(f"Max offdiag Phiᵀ M Phi     : {result.max_mass_offdiag:.3e}")
        print(f"Max |diag(Phiᵀ M Phi)-1|   : {result.max_mass_normalization_error:.3e}")
        print(f"Max offdiag Phiᵀ K Phi     : {result.max_stiffness_offdiag:.3e}")
        print(f"Max offdiag Phiᵀ K Phi rel : {result.max_stiffness_offdiag_rel:.3e}")
        print(f"Max |phiᵀKphi-λ|           : {result.max_stiffness_consistency_error:.3e}")
        print(f"Max |phiᵀKphi-λ| rel       : {result.max_stiffness_consistency_error_rel:.3e}")
        print(f"Max résidu relatif         : {result.max_residual_rel:.3e}")
