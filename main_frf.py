# main_frf.py

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from models import Material, BlackHole, Plate, MeshConfig, HarmonicPointForce, Sensor
from fem import FEMModel
from solvers import FRFSolver, RayleighDamping


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


def build_excitation() -> HarmonicPointForce:
    return HarmonicPointForce(
        x=0.10,
        y=0.10,
        amplitude=1.0,
        frequency_start=80.0,
        frequency_end=600.0,
        n_points=80,
        phase_deg=0.0,
        direction="w",
    )


def build_sensor() -> Sensor:
    return Sensor(
        x=0.35,
        y=0.25,
        name="S1",
        response_type="displacement",
    )


def solve_case(use_black_hole: bool):
    plate = build_plate()
    mesh_config = build_mesh_config()
    excitation = build_excitation()
    sensor = build_sensor()

    print("\nConstruction du modèle...", "avec VABH" if use_black_hole else "uniforme")
    model = FEMModel(
        plate=plate,
        mesh_config=mesh_config,
        use_black_hole=use_black_hole,
        refine_with_black_hole_region=True,
    )
    model.build()

    print("Modèle construit. Calcul de l'amortissement...")
    damping = RayleighDamping.from_modal_damping_ratio(
        zeta=0.01,
        freq1_hz=140.0,
        freq2_hz=500.0,
    )

    print("Lancement du calcul FRF...")
    frf_solver = FRFSolver(model)
    result = frf_solver.solve(
        excitation=excitation,
        sensor=sensor,
        damping=damping,
    )

    print("Calcul FRF terminé.")
    print("\n=== Cas", "avec VABH ===" if use_black_hole else "uniforme ===")
    print(f"Noeud excitation : {result.excitation_node_id} | écart géométrique = {result.excitation_distance_m*1e3:.3f} mm")
    print(f"Noeud capteur    : {result.sensor_node_id} | écart géométrique = {result.sensor_distance_m*1e3:.3f} mm")

    return result


def plot_frf(result_ref, result_bh) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(result_ref.frequencies_hz, result_ref.magnitude, label="Plaque uniforme")
    ax.semilogy(result_bh.frequencies_hz, result_bh.magnitude, label="Plaque avec VABH")
    ax.set_xlabel("Fréquence [Hz]")
    ax.set_ylabel("|FRF| [m/N]")
    ax.set_title("Comparaison des FRF en déplacement")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(result_ref.frequencies_hz, result_ref.phase_deg, label="Plaque uniforme")
    ax.plot(result_bh.frequencies_hz, result_bh.phase_deg, label="Plaque avec VABH")
    ax.set_xlabel("Fréquence [Hz]")
    ax.set_ylabel("Phase [deg]")
    ax.set_title("Phase de la FRF")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()


def main() -> None:
    result_ref = solve_case(use_black_hole=False)
    result_bh = solve_case(use_black_hole=True)

    plot_frf(result_ref, result_bh)
    plt.show()


if __name__ == "__main__":
    main()