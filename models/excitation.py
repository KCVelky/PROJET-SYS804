# models/excitation.py

from __future__ import annotations

from dataclasses import dataclass

from utils.validators import ensure_non_negative, ensure_positive


@dataclass
class HarmonicPointForce:
    """
    Force ponctuelle harmonique :
    F(t) = F0 * sin(2*pi*f*t) ou représentation complexe en fréquentiel.
    Paramètres en SI :
    - x, y : m
    - amplitude : N
    - frequency_start, frequency_end : Hz
    """

    x: float
    y: float
    amplitude: float
    frequency_start: float
    frequency_end: float
    n_points: int = 1000
    phase_deg: float = 0.0
    direction: str = "w"

    def __post_init__(self) -> None:
        ensure_non_negative(self.x, "Position x de la force")
        ensure_non_negative(self.y, "Position y de la force")
        ensure_positive(self.amplitude, "Amplitude de la force")
        ensure_non_negative(self.frequency_start, "Fréquence de départ")
        ensure_positive(self.frequency_end, "Fréquence de fin")

        if self.frequency_end <= self.frequency_start:
            raise ValueError("La fréquence de fin doit être supérieure à la fréquence de départ.")

        if self.n_points < 2:
            raise ValueError("n_points doit être >= 2.")

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "amplitude": self.amplitude,
            "frequency_start": self.frequency_start,
            "frequency_end": self.frequency_end,
            "n_points": self.n_points,
            "phase_deg": self.phase_deg,
            "direction": self.direction,
        }