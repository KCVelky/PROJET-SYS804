# models/sensor.py

from __future__ import annotations

from dataclasses import dataclass

from utils.validators import ensure_non_negative, ensure_str_in


_ALLOWED_RESPONSE_TYPES = (
    "displacement",
    "velocity",
    "acceleration",
)


@dataclass
class Sensor:
    """
    Capteur ponctuel de réponse.
    Paramètres en SI :
    - x, y : m
    """

    x: float
    y: float
    name: str = "S1"
    response_type: str = "displacement"

    def __post_init__(self) -> None:
        ensure_non_negative(self.x, "Position x du capteur")
        ensure_non_negative(self.y, "Position y du capteur")
        ensure_str_in(self.response_type, "Type de réponse du capteur", _ALLOWED_RESPONSE_TYPES)

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "name": self.name,
            "response_type": self.response_type,
        }