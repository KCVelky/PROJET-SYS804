# main_compare_modal.py

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from models import Material, BlackHole, Plate, MeshConfig
from fem import FEMModel
from solvers import ModalSolver


def build_plate() -> Plate:
    material = Material.aluminum()

    black_hole = BlackHole(
        xc=0.25,
        yc=0.20,
        radius=0.06,
        truncation_radius=0.005,
        residual_thickness=0.0003,
        exponent=2.0,
        enabled=True,
    )

    plate = Plate(
        length_x=0.50,
        length_y=0.40,
        thickness=0.002,
        material=material,
        boundary_condition="simply_supported",
        black_hole=black_hole,
        name="Plate with or without VABH",
    )

    plate.validate_black_hole_inside()
    return plate


def build_mesh_config() -> MeshConfig:
    return MeshConfig(
        element_size=0.02,
        refine_near_black_hole=True,
        refinement_radius=0.08,
        refinement_element_size=0.005,
        element_type="tri3",
    )


def solve_case(plate: Plate, mesh_config: MeshConfig, use_black_hole: bool, n_modes: int = 6):
    model = FEMModel(
        plate=plate,
        mesh_config=mesh_config,
        use_black_hole=use_black_hole,
        refine_with_black_hole_region=True,
    )
    model.build()

    solver = ModalSolver(model)
    freqs, modes = solver.solve(n_modes=n_modes)

    return model, freqs, modes


def print_comparison(freqs_ref: np.ndarray, freqs_bh: np.ndarray) -> None:
    n = min(len(freqs_ref), len(freqs_bh))
    print("\n=== Comparaison modale : plaque uniforme vs plaque avec VABH ===")
    print(f"{'Mode':>4} | {'Référence [Hz]':>15} | {'VABH [Hz]':>12} | {'Écart [%]':>10}")
    print("-" * 52)

    for i in range(n):
        delta_pct = 100.0 * (freqs_bh[i] - freqs_ref[i]) / freqs_ref[i]
        print(f"{i+1:>4} | {freqs_ref[i]:>15.3f} | {freqs_bh[i]:>12.3f} | {delta_pct:>10.3f}")


def plot_frequency_comparison(freqs_ref: np.ndarray, freqs_bh: np.ndarray) -> None:
    n = min(len(freqs_ref), len(freqs_bh))
    modes = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(modes, freqs_ref[:n], "o-", label="Plaque uniforme")
    ax.plot(modes, freqs_bh[:n], "s-", label="Plaque avec VABH")
    ax.set_xlabel("Numéro de mode")
    ax.set_ylabel("Fréquence propre [Hz]")
    ax.set_title("Comparaison des fréquences propres")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()


def plot_relative_shift(freqs_ref: np.ndarray, freqs_bh: np.ndarray) -> None:
    n = min(len(freqs_ref), len(freqs_bh))
    modes = np.arange(1, n + 1)
    shift_pct = 100.0 * (freqs_bh[:n] - freqs_ref[:n]) / freqs_ref[:n]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(modes, shift_pct)
    ax.set_xlabel("Numéro de mode")
    ax.set_ylabel("Variation relative [%]")
    ax.set_title("Impact relatif du trou noir vibratoire sur les fréquences propres")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def main() -> None:
    plate = build_plate()
    mesh_config = build_mesh_config()

    _, freqs_ref, _ = solve_case(
        plate=plate,
        mesh_config=mesh_config,
        use_black_hole=False,
        n_modes=6,
    )

    _, freqs_bh, _ = solve_case(
        plate=plate,
        mesh_config=mesh_config,
        use_black_hole=True,
        n_modes=6,
    )

    print_comparison(freqs_ref, freqs_bh)
    plot_frequency_comparison(freqs_ref, freqs_bh)
    plot_relative_shift(freqs_ref, freqs_bh)

    plt.show()


if __name__ == "__main__":
    main()