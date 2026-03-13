# solvers/frf_solver.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fem.fem_model import FEMModel
from models.excitation import HarmonicPointForce
from models.sensor import Sensor
from solvers.damping import RayleighDamping
from solvers.harmonic_solver import HarmonicSolver


@dataclass
class FRFResult:
    frequencies_hz: np.ndarray
    response_complex: np.ndarray
    frf_complex: np.ndarray
    excitation_node_id: int
    sensor_node_id: int
    excitation_distance_m: float
    sensor_distance_m: float
    response_type: str

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.frf_complex)

    @property
    def phase_deg(self) -> np.ndarray:
        return np.angle(self.frf_complex, deg=True)


class FRFSolver:
    """
    Calcule une FRF entre une excitation harmonique ponctuelle et un capteur.
    """

    def __init__(self, fem_model: FEMModel) -> None:
        self.model = fem_model
        self.harmonic_solver = HarmonicSolver(fem_model)

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

    def _get_sensor_dof(self, sensor: Sensor) -> tuple[int, float]:
        node_id, distance = self.model.find_nearest_node(sensor.x, sensor.y)

        # Le capteur lit d'abord la déflexion w,
        # puis on convertit éventuellement en vitesse / accélération.
        dof_id = self.model.dof_index(node_id, dof_name="w")

        if not self.model.is_free_dof(dof_id):
            raise ValueError(
                "Le capteur est positionné sur un ddl bloqué par les conditions aux limites."
            )

        return dof_id, distance

    def solve(
        self,
        excitation: HarmonicPointForce,
        sensor: Sensor,
        damping: RayleighDamping | None = None,
    ) -> FRFResult:
        frequencies_hz = np.linspace(
            excitation.frequency_start,
            excitation.frequency_end,
            excitation.n_points,
        )

        force_vector, F_complex, excitation_node_id, excitation_distance = self._build_force_vector(excitation)
        sensor_dof_id, sensor_distance = self._get_sensor_dof(sensor)

        w_response = self.harmonic_solver.response_at_dofs(
            frequencies_hz=frequencies_hz,
            force_vector_full=force_vector,
            output_dofs_full=np.array([sensor_dof_id], dtype=int),
            damping=damping,
        )[0]

        omega = 2.0 * np.pi * frequencies_hz

        if sensor.response_type == "displacement":
            response_complex = w_response
        elif sensor.response_type == "velocity":
            response_complex = 1j * omega * w_response
        elif sensor.response_type == "acceleration":
            response_complex = -(omega ** 2) * w_response
        else:
            raise ValueError(f"Type de réponse capteur non géré : {sensor.response_type}")

        frf_complex = response_complex / F_complex

        sensor_node_id, _ = self.model.find_nearest_node(sensor.x, sensor.y)

        return FRFResult(
            frequencies_hz=frequencies_hz,
            response_complex=response_complex,
            frf_complex=frf_complex,
            excitation_node_id=excitation_node_id,
            sensor_node_id=sensor_node_id,
            excitation_distance_m=excitation_distance,
            sensor_distance_m=sensor_distance,
            response_type=sensor.response_type,
        )