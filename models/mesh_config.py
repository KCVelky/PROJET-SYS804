# models/mesh_config.py

from __future__ import annotations

from dataclasses import dataclass

from utils.validators import ensure_positive, ensure_non_negative


@dataclass
class MeshConfig:
    """
    Paramètres de maillage.
    Toutes les tailles sont en m.
    """

    element_size: float
    refine_near_black_hole: bool = True
    refinement_radius: float = 0.08
    refinement_element_size: float = 0.005
    element_type: str = "tri3"

    def __post_init__(self) -> None:
        ensure_positive(self.element_size, "Taille globale des éléments")
        ensure_non_negative(self.refinement_radius, "Rayon de raffinement")
        ensure_positive(self.refinement_element_size, "Taille locale raffinée")

        if self.refinement_element_size > self.element_size:
            raise ValueError(
                "La taille raffinée doit être inférieure ou égale à la taille globale."
            )

    def to_dict(self) -> dict:
        return {
            "element_size": self.element_size,
            "refine_near_black_hole": self.refine_near_black_hole,
            "refinement_radius": self.refinement_radius,
            "refinement_element_size": self.refinement_element_size,
            "element_type": self.element_type,
        }