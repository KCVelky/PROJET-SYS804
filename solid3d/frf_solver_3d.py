from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solvers.damping import RayleighDamping
from solvers.harmonic_solver import HarmonicSolver
from solid3d.fem_model_3d import Solid3DFEMModel
from solid3d.probes_3d import (
    HarmonicPointForce3D,
    PointSensor3D,
    build_force_vector_3d,
    get_sensor_dof_3d,
)


@dataclass
class FRFResult3D:
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

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.frf_complex)

    @property
    def phase_deg(self) -> np.ndarray:
        return np.angle(self.frf_complex, deg=True)


class FRFSolver3D:
    """
    FRF directe 3D solide :
    (K + j*w*C - w^2*M) q = f

    La lecture se fait sur une composante x, y ou z d'un capteur ponctuel 3D.
    """

    def __init__(self, fem_model: Solid3DFEMModel, verbose: bool = True) -> None:
        self.model = fem_model
        self.verbose = verbose
        self.harmonic_solver = HarmonicSolver(fem_model)

    def solve(
        self,
        excitation: HarmonicPointForce3D,
        sensor: PointSensor3D,
        damping: RayleighDamping | None = None,
    ) -> FRFResult3D:
        if self.model.Kff is None or self.model.Mff is None or self.model.free_dofs is None:
            raise RuntimeError("Le modèle 3D doit être construit avant le calcul FRF direct.")

        frequencies_hz = np.linspace(
            excitation.frequency_start,
            excitation.frequency_end,
            excitation.n_points,
        )
        if np.any(frequencies_hz <= 0.0):
            raise ValueError("Toutes les fréquences FRF 3D doivent être strictement positives.")

        force_vector, f_complex, excitation_node_id, excitation_distance, excitation_dof_id = (
            build_force_vector_3d(self.model, excitation)
        )
        sensor_dof_id, sensor_node_id, sensor_distance = get_sensor_dof_3d(self.model, sensor)

        responses = self.harmonic_solver.response_at_dofs(
            frequencies_hz=frequencies_hz,
            force_vector_full=force_vector,
            output_dofs_full=np.array([sensor_dof_id], dtype=int),
            damping=damping,
        )
        displacement_complex = responses[0, :]

        omega = 2.0 * np.pi * frequencies_hz
        if sensor.response_type == "displacement":
            response_complex = displacement_complex
        elif sensor.response_type == "velocity":
            response_complex = 1j * omega * displacement_complex
        elif sensor.response_type == "acceleration":
            response_complex = -(omega**2) * displacement_complex
        else:
            raise ValueError(f"Type de réponse 3D non géré : {sensor.response_type}")

        frf_complex = response_complex / f_complex

        return FRFResult3D(
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
        )
