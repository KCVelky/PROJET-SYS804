# solvers/modal_frf_solver.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fem.fem_model import FEMModel
from models.excitation import HarmonicPointForce
from models.sensor import Sensor
from solvers.modal_solver import ModalSolver, ModalBasis


@dataclass
class ModalFRFResult:
    frequencies_hz: np.ndarray
    response_complex: np.ndarray
    frf_complex: np.ndarray
    excitation_node_id: int
    sensor_node_id: int
    excitation_distance_m: float
    sensor_distance_m: float
    response_type: str
    n_modes_used: int
    retained_mode_frequencies_hz: np.ndarray

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.frf_complex)

    @property
    def phase_deg(self) -> np.ndarray:
        return np.angle(self.frf_complex, deg=True)


class ModalFRFSolver:
    """
    FRF par superposition modale :
    q(omega) ≈ Phi * eta(omega)

    Hypothèses :
    - base modale tronquée
    - modes normalisés par rapport à la masse
    - amortissement modal diagonal
    """

    def __init__(self, fem_model: FEMModel) -> None:
        self.model = fem_model
        self.modal_solver = ModalSolver(fem_model)

    def _build_force_vector(
        self,
        excitation: HarmonicPointForce,
    ) -> tuple[np.ndarray, complex, int, float]:
        node_id, distance = self.model.find_nearest_node(excitation.x, excitation.y)

        dof_name = excitation.direction
        dof_id = self.model.dof_index(node_id, dof_name=dof_name)

        if not self.model.is_free_dof(dof_id):
            raise ValueError(
                "La force est appliquée sur un ddl bloqué par les conditions aux limites."
            )

        F_complex = excitation.amplitude * np.exp(1j * np.deg2rad(excitation.phase_deg))

        force_vector = np.zeros(self.model.n_dofs, dtype=complex)
        force_vector[dof_id] = F_complex

        return force_vector, F_complex, node_id, distance

    def _get_sensor_dof(self, sensor: Sensor) -> tuple[int, int, float]:
        node_id, distance = self.model.find_nearest_node(sensor.x, sensor.y)

        dof_id = self.model.dof_index(node_id, dof_name="w")

        if not self.model.is_free_dof(dof_id):
            raise ValueError(
                "Le capteur est positionné sur un ddl bloqué par les conditions aux limites."
            )

        return dof_id, node_id, distance

    def _full_to_free_position(self, dof_id: int) -> int:
        if self.model.free_dofs is None:
            raise RuntimeError("Les ddl libres ne sont pas disponibles.")

        pos = int(np.searchsorted(self.model.free_dofs, dof_id))
        if pos >= len(self.model.free_dofs) or self.model.free_dofs[pos] != dof_id:
            raise ValueError(f"Le ddl {dof_id} n'est pas libre.")
        return pos

    @staticmethod
    def _prepare_modal_damping(
        damping_ratio: float | np.ndarray,
        n_modes: int,
    ) -> np.ndarray:
        if np.isscalar(damping_ratio):
            zeta = np.full(n_modes, float(damping_ratio), dtype=float)
        else:
            zeta = np.asarray(damping_ratio, dtype=float)
            if zeta.shape[0] != n_modes:
                raise ValueError("Le tableau des amortissements modaux n'a pas la bonne taille.")

        if np.any(zeta < 0.0):
            raise ValueError("Les amortissements modaux doivent être positifs ou nuls.")

        return zeta

    def solve(
        self,
        excitation: HarmonicPointForce,
        sensor: Sensor,
        n_modes: int = 30,
        damping_ratio: float | np.ndarray = 0.01,
        modal_basis: ModalBasis | None = None,
    ) -> ModalFRFResult:
        if self.model.Kff is None or self.model.Mff is None or self.model.free_dofs is None:
            raise RuntimeError("Le modèle EF doit être construit avant le calcul FRF modal.")

        frequencies_hz = np.linspace(
            excitation.frequency_start,
            excitation.frequency_end,
            excitation.n_points,
        )
        if np.any(frequencies_hz <= 0.0):
            raise ValueError("Les fréquences doivent être strictement positives.")

        if modal_basis is None:
            modal_basis = self.modal_solver.solve_basis(n_modes=n_modes)

        n_available = modal_basis.modes_free.shape[1]
        n_used = min(n_modes, n_available)

        Phi = modal_basis.modes_free[:, :n_used]                     # (n_free, n_used)
        wn = modal_basis.omegas_rad_s[:n_used]                      # (n_used,)
        zeta = self._prepare_modal_damping(damping_ratio, n_used)   # (n_used,)

        force_vector, F_complex, excitation_node_id, excitation_distance = self._build_force_vector(excitation)
        sensor_dof_id, sensor_node_id, sensor_distance = self._get_sensor_dof(sensor)

        force_free = force_vector[self.model.free_dofs]

        # forces modales généralisées : p_r = phi_r^T f
        modal_forces = Phi.conj().T @ force_free                    # (n_used,)

        # lecture au capteur
        sensor_pos_free = self._full_to_free_position(sensor_dof_id)
        phi_sensor = Phi[sensor_pos_free, :]                        # (n_used,)

        omega = 2.0 * np.pi * frequencies_hz                        # (nf,)
        wn_col = wn[:, None]                                        # (n_used, 1)
        omega_row = omega[None, :]                                  # (1, nf)
        zeta_col = zeta[:, None]                                    # (n_used, 1)

        denom = wn_col**2 - omega_row**2 + 2j * zeta_col * wn_col * omega_row
        eta = modal_forces[:, None] / denom                         # (n_used, nf)

        # réponse transverse w au capteur
        w_response = phi_sensor @ eta                               # (nf,)

        if sensor.response_type == "displacement":
            response_complex = w_response
        elif sensor.response_type == "velocity":
            response_complex = 1j * omega * w_response
        elif sensor.response_type == "acceleration":
            response_complex = -(omega**2) * w_response
        else:
            raise ValueError(f"Type de réponse capteur non géré : {sensor.response_type}")

        frf_complex = response_complex / F_complex

        return ModalFRFResult(
            frequencies_hz=frequencies_hz,
            response_complex=response_complex,
            frf_complex=frf_complex,
            excitation_node_id=excitation_node_id,
            sensor_node_id=sensor_node_id,
            excitation_distance_m=excitation_distance,
            sensor_distance_m=sensor_distance,
            response_type=sensor.response_type,
            n_modes_used=n_used,
            retained_mode_frequencies_hz=modal_basis.frequencies_hz[:n_used],
        )