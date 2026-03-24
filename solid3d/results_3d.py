from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModalValidationPerMode3D:
    mode_number: int
    frequency_hz: float
    eigenvalue: float
    modal_mass: float
    mass_normalization_error: float
    stiffness_consistency_error: float
    stiffness_consistency_error_rel: float
    residual_norm: float
    residual_rel: float


@dataclass
class ModalValidationResult3D:
    n_nodes: int
    n_elements: int
    n_dofs: int
    n_free_dofs: int
    n_constrained_dofs: int
    n_used_nodes: int
    n_unused_nodes: int
    frequencies_hz: np.ndarray
    mass_orthogonality_matrix: np.ndarray
    stiffness_orthogonality_matrix: np.ndarray
    per_mode: list[ModalValidationPerMode3D]

    @property
    def max_mass_offdiag(self) -> float:
        if self.mass_orthogonality_matrix.size == 0:
            return 0.0
        offdiag = self.mass_orthogonality_matrix - np.diag(np.diag(self.mass_orthogonality_matrix))
        return float(np.max(np.abs(offdiag)))

    @property
    def max_mass_normalization_error(self) -> float:
        if self.mass_orthogonality_matrix.size == 0:
            return 0.0
        diag = np.diag(self.mass_orthogonality_matrix)
        return float(np.max(np.abs(diag - 1.0)))

    @property
    def max_stiffness_offdiag(self) -> float:
        if self.stiffness_orthogonality_matrix.size == 0:
            return 0.0
        offdiag = self.stiffness_orthogonality_matrix - np.diag(np.diag(self.stiffness_orthogonality_matrix))
        return float(np.max(np.abs(offdiag)))

    @property
    def max_stiffness_offdiag_rel(self) -> float:
        if self.stiffness_orthogonality_matrix.size == 0:
            return 0.0
        diag = np.abs(np.diag(self.stiffness_orthogonality_matrix))
        scale = max(1.0, float(np.max(diag)))
        return self.max_stiffness_offdiag / scale

    @property
    def max_stiffness_consistency_error(self) -> float:
        if not self.per_mode:
            return 0.0
        return float(max(abs(item.stiffness_consistency_error) for item in self.per_mode))

    @property
    def max_stiffness_consistency_error_rel(self) -> float:
        if not self.per_mode:
            return 0.0
        return float(max(abs(item.stiffness_consistency_error_rel) for item in self.per_mode))

    @property
    def max_residual_rel(self) -> float:
        if not self.per_mode:
            return 0.0
        return float(max(abs(item.residual_rel) for item in self.per_mode))
