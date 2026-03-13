# solvers/harmonic_solver.py

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

from fem.fem_model import FEMModel
from solvers.damping import RayleighDamping


class HarmonicSolver:
    """
    Solveur harmonique direct :
    (K + j*w*C - w^2*M) q = f
    """

    def __init__(self, fem_model: FEMModel) -> None:
        self.model = fem_model

    def _map_full_dofs_to_free(self, dofs_full: np.ndarray) -> np.ndarray:
        if self.model.free_dofs is None:
            raise RuntimeError("Les ddl libres ne sont pas disponibles.")

        positions = np.searchsorted(self.model.free_dofs, dofs_full)

        valid = (
            (positions >= 0)
            & (positions < len(self.model.free_dofs))
            & (self.model.free_dofs[positions] == dofs_full)
        )

        if not np.all(valid):
            bad_dofs = dofs_full[~valid]
            raise ValueError(
                f"Certains ddl demandés sont bloqués par les conditions aux limites : {bad_dofs.tolist()}"
            )

        return positions

    def response_at_dofs(
        self,
        frequencies_hz: np.ndarray,
        force_vector_full: np.ndarray,
        output_dofs_full: np.ndarray,
        damping: RayleighDamping | None = None,
    ) -> np.ndarray:
        """
        Retourne les réponses complexes aux ddl demandés.

        Paramètres :
        - frequencies_hz    : tableau (nf,)
        - force_vector_full : tableau global complet (n_dofs,)
        - output_dofs_full  : tableau des ddl globaux à observer

        Retour :
        - responses : tableau complexe (n_outputs, nf)
        """
        if self.model.Kff is None or self.model.Mff is None or self.model.free_dofs is None:
            raise RuntimeError("Le modèle EF doit être construit avant l'analyse harmonique.")

        frequencies_hz = np.asarray(frequencies_hz, dtype=float)
        if np.any(frequencies_hz <= 0.0):
            raise ValueError("Toutes les fréquences doivent être strictement positives.")

        force_vector_full = np.asarray(force_vector_full, dtype=complex)
        output_dofs_full = np.asarray(output_dofs_full, dtype=int)

        if force_vector_full.shape[0] != self.model.n_dofs:
            raise ValueError("La taille du vecteur de force global ne correspond pas au nombre de ddl.")

        force_vector_free = force_vector_full[self.model.free_dofs]
        output_dofs_free = self._map_full_dofs_to_free(output_dofs_full)

        Kff = self.model.Kff.tocsc()
        Mff = self.model.Mff.tocsc()

        if damping is None:
            Cff = csr_matrix(Kff.shape, dtype=float)
        else:
            Cff = damping.matrix(self.model.Mff, self.model.Kff).tocsc()

        n_outputs = len(output_dofs_full)
        n_freqs = len(frequencies_hz)
        responses = np.zeros((n_outputs, n_freqs), dtype=complex)

        for i, freq_hz in enumerate(frequencies_hz):
            if i % 10 == 0 or i == len(frequencies_hz) - 1:
                print(f"FRF en cours : {i+1}/{len(frequencies_hz)} fréquences | f = {freq_hz:.2f} Hz")

            omega = 2.0 * np.pi * freq_hz
            dynamic_stiffness = Kff + 1j * omega * Cff - (omega ** 2) * Mff
            q_free = spsolve(dynamic_stiffness, force_vector_free)
            responses[:, i] = q_free[output_dofs_free]
        return responses