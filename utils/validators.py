# utils/validators.py

from __future__ import annotations


def ensure_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} doit être strictement positif. Reçu : {value}")


def ensure_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} doit être positif ou nul. Reçu : {value}")


def ensure_in_range(value: float, name: str, vmin: float, vmax: float) -> None:
    if not (vmin <= value <= vmax):
        raise ValueError(
            f"{name} doit appartenir à l'intervalle [{vmin}, {vmax}]. Reçu : {value}"
        )


def ensure_str_in(value: str, name: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(
            f"{name} doit appartenir à {allowed}. Reçu : '{value}'"
        )