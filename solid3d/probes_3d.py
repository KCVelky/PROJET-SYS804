from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solid3d.fem_model_3d import Solid3DFEMModel
from utils.validators import ensure_non_negative, ensure_positive, ensure_str_in


_ALLOWED_DIRECTIONS_3D = ("x", "y", "z")
_ALLOWED_RESPONSE_TYPES_3D = (
    "displacement",
    "velocity",
    "acceleration",
)


@dataclass
class HarmonicPointForce3D:
    """
    Force ponctuelle harmonique 3D.

    Convention :
    - x, y, z en m
    - amplitude en N
    - fréquences en Hz
    - direction parmi x, y, z
    """

    x: float
    y: float
    z: float
    amplitude: float
    frequency_start: float
    frequency_end: float
    n_points: int = 400
    phase_deg: float = 0.0
    direction: str = "z"

    def __post_init__(self) -> None:
        ensure_non_negative(self.x, "Position x de la force 3D")
        ensure_non_negative(self.y, "Position y de la force 3D")
        ensure_non_negative(self.z, "Position z de la force 3D")
        ensure_positive(self.amplitude, "Amplitude de la force 3D")
        ensure_non_negative(self.frequency_start, "Fréquence de départ 3D")
        ensure_positive(self.frequency_end, "Fréquence de fin 3D")
        ensure_str_in(self.direction, "Direction de la force 3D", _ALLOWED_DIRECTIONS_3D)

        if self.frequency_end <= self.frequency_start:
            raise ValueError("La fréquence de fin doit être supérieure à la fréquence de départ.")
        if self.n_points < 2:
            raise ValueError("n_points doit être >= 2.")

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "amplitude": self.amplitude,
            "frequency_start": self.frequency_start,
            "frequency_end": self.frequency_end,
            "n_points": self.n_points,
            "phase_deg": self.phase_deg,
            "direction": self.direction,
        }


@dataclass
class PointSensor3D:
    """
    Capteur ponctuel 3D.

    direction : composante observée parmi x, y, z
    response_type : déplacement, vitesse ou accélération
    """

    x: float
    y: float
    z: float
    name: str = "S1"
    direction: str = "z"
    response_type: str = "displacement"

    def __post_init__(self) -> None:
        ensure_non_negative(self.x, "Position x du capteur 3D")
        ensure_non_negative(self.y, "Position y du capteur 3D")
        ensure_non_negative(self.z, "Position z du capteur 3D")
        ensure_str_in(self.direction, "Direction du capteur 3D", _ALLOWED_DIRECTIONS_3D)
        ensure_str_in(self.response_type, "Type de réponse du capteur 3D", _ALLOWED_RESPONSE_TYPES_3D)

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "name": self.name,
            "direction": self.direction,
            "response_type": self.response_type,
        }


def dof_offset_3d(direction: str) -> int:
    ensure_str_in(direction, "Direction ddl 3D", _ALLOWED_DIRECTIONS_3D)
    return {"x": 0, "y": 1, "z": 2}[direction]


def dof_index_3d(node_id: int, direction: str) -> int:
    return 3 * int(node_id) + dof_offset_3d(direction)


def is_free_dof_3d(model: Solid3DFEMModel, dof_id: int) -> bool:
    if model.free_dofs is None:
        raise RuntimeError("Les ddl libres 3D ne sont pas disponibles.")

    pos = int(np.searchsorted(model.free_dofs, dof_id))
    return pos < len(model.free_dofs) and int(model.free_dofs[pos]) == int(dof_id)


def build_force_vector_3d(
    model: Solid3DFEMModel,
    excitation: HarmonicPointForce3D,
) -> tuple[np.ndarray, complex, int, float, int]:
    if model.mesh is None:
        raise RuntimeError("Le maillage 3D doit être construit avant la force ponctuelle.")

    node_id, distance = model.find_nearest_node(excitation.x, excitation.y, excitation.z)
    dof_id = dof_index_3d(node_id, excitation.direction)

    if not is_free_dof_3d(model, dof_id):
        raise ValueError(
            "La force 3D est appliquée sur un ddl bloqué par les conditions aux limites."
        )

    f_complex = excitation.amplitude * np.exp(1j * np.deg2rad(excitation.phase_deg))
    force_vector = np.zeros(model.n_dofs, dtype=complex)
    force_vector[dof_id] = f_complex
    return force_vector, complex(f_complex), int(node_id), float(distance), int(dof_id)


def get_sensor_dof_3d(
    model: Solid3DFEMModel,
    sensor: PointSensor3D,
) -> tuple[int, int, float]:
    if model.mesh is None:
        raise RuntimeError("Le maillage 3D doit être construit avant la lecture capteur.")

    node_id, distance = model.find_nearest_node(sensor.x, sensor.y, sensor.z)
    dof_id = dof_index_3d(node_id, sensor.direction)

    if not is_free_dof_3d(model, dof_id):
        raise ValueError(
            "Le capteur 3D est positionné sur un ddl bloqué par les conditions aux limites."
        )

    return int(dof_id), int(node_id), float(distance)
