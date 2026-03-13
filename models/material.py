# models/material.py

from __future__ import annotations

from dataclasses import dataclass

from utils.validators import ensure_positive, ensure_in_range


@dataclass
class Material:
    """
    Matériau isotrope linéaire élastique.
    Toutes les unités sont en SI :
    - E en Pa
    - rho en kg/m^3
    """

    young_modulus: float
    poisson_ratio: float
    density: float
    name: str = "Generic material"

    def __post_init__(self) -> None:
        ensure_positive(self.young_modulus, "Module d'Young")
        ensure_positive(self.density, "Masse volumique")
        ensure_in_range(self.poisson_ratio, "Coefficient de Poisson", 0.0, 0.499)

    @classmethod
    def aluminum(cls) -> "Material":
        """
        Valeurs cohérentes avec ton projet :
        E = 69 GPa, nu = 0.33, rho = 2700 kg/m^3
        """
        return cls(
            young_modulus=69e9,
            poisson_ratio=0.33,
            density=2700.0,
            name="Aluminum"
        )

    @property
    def E(self) -> float:
        return self.young_modulus

    @property
    def nu(self) -> float:
        return self.poisson_ratio

    @property
    def rho(self) -> float:
        return self.density

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "young_modulus": self.young_modulus,
            "poisson_ratio": self.poisson_ratio,
            "density": self.density,
        }