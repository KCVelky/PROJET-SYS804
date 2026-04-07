from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solid3d.fem_model_3d import Solid3DFEMModel
from solid3d.modal_solver_3d import ModalBasis3D, ModalSolver3D
from solid3d.probes_3d import (
    HarmonicPointForce3D,
    PointSensor3D,
    build_force_vector_3d,
    get_sensor_dof_3d,
)


@dataclass
class ModalFRFResult3D:
    frequencies_hz: np.ndarray
    response_complex: np.ndarray
    frf_complex: np.ndarray
    excitation_node_id: int
    sensor_node_id: int
    excitation_dof_id: int
    sensor_dof_id: int
    excitation_distance_m: float
    sensor_distance_m: float
    excitation_direction: str
    sensor_direction: str
    response_type: str
    n_modes_used: int
    retained_mode_frequencies_hz: np.ndarray

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.frf_complex)

    @property
    def phase_deg(self) -> np.ndarray:
        return np.angle(self.frf_complex, deg=True)


class ModalFRFSolver3D:
    """
    FRF 3D par superposition modale :
    q(omega) ≈ Phi * eta(omega)

    Hypothèses :
    - base modale tronquée
    - modes mass-normalisés
    - amortissement modal diagonal
    """

    def __init__(self, fem_model: Solid3DFEMModel, verbose: bool = True) -> None:
        self.model = fem_model
        self.verbose = verbose
        self.modal_solver = ModalSolver3D(fem_model, verbose=verbose)

    def _full_to_free_position(self, dof_id: int) -> int:
        if self.model.free_dofs is None:
            raise RuntimeError("Les ddl libres 3D ne sont pas disponibles.")

        pos = int(np.searchsorted(self.model.free_dofs, dof_id))
        if pos >= len(self.model.free_dofs) or int(self.model.free_dofs[pos]) != int(dof_id):
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
        excitation: HarmonicPointForce3D,
        sensor: PointSensor3D,
        n_modes: int = 30,
        damping_ratio: float | np.ndarray = 0.01,
        modal_basis: ModalBasis3D | None = None,
    ) -> ModalFRFResult3D:
        if self.model.Kff is None or self.model.Mff is None or self.model.free_dofs is None:
            raise RuntimeError("Le modèle 3D doit être construit avant le calcul FRF modal.")

        frequencies_hz = np.linspace(
            excitation.frequency_start,
            excitation.frequency_end,
            excitation.n_points,
        )
        if np.any(frequencies_hz <= 0.0):
            raise ValueError("Toutes les fréquences FRF 3D doivent être strictement positives.")

        if modal_basis is None:
            modal_basis = self.modal_solver.solve_basis(n_modes=n_modes)

        n_available = modal_basis.modes_free.shape[1]
        n_used = min(n_modes, n_available)

        Phi = modal_basis.modes_free[:, :n_used]
        wn = modal_basis.omegas_rad_s[:n_used]
        zeta = self._prepare_modal_damping(damping_ratio, n_used)

        force_vector, f_complex, excitation_node_id, excitation_distance, excitation_dof_id = (
            build_force_vector_3d(self.model, excitation)
        )
        sensor_dof_id, sensor_node_id, sensor_distance = get_sensor_dof_3d(self.model, sensor)

        force_free = force_vector[self.model.free_dofs]

        modal_forces = Phi.conj().T @ force_free

        sensor_pos_free = self._full_to_free_position(sensor_dof_id)
        phi_sensor = Phi[sensor_pos_free, :]

        omega = 2.0 * np.pi * frequencies_hz
        wn_col = wn[:, None]
        omega_row = omega[None, :]
        zeta_col = zeta[:, None]

        denom = wn_col**2 - omega_row**2 + 2j * zeta_col * wn_col * omega_row
        eta = modal_forces[:, None] / denom

        displacement_complex = phi_sensor @ eta

        if sensor.response_type == "displacement":
            response_complex = displacement_complex
        elif sensor.response_type == "velocity":
            response_complex = 1j * omega * displacement_complex
        elif sensor.response_type == "acceleration":
            response_complex = -(omega**2) * displacement_complex
        else:
            raise ValueError(f"Type de réponse 3D non géré : {sensor.response_type}")

        frf_complex = response_complex / f_complex

        return ModalFRFResult3D(
            frequencies_hz=frequencies_hz,
            response_complex=response_complex,
            frf_complex=frf_complex,
            excitation_node_id=excitation_node_id,
            sensor_node_id=sensor_node_id,
            excitation_dof_id=excitation_dof_id,
            sensor_dof_id=sensor_dof_id,
            excitation_distance_m=excitation_distance,
            sensor_distance_m=sensor_distance,
            excitation_direction=excitation.direction,
            sensor_direction=sensor.direction,
            response_type=sensor.response_type,
            n_modes_used=n_used,
            retained_mode_frequencies_hz=modal_basis.frequencies_hz[:n_used],
        )