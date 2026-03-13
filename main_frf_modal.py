# main_frf_modal.py

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from models import Material, BlackHole, Plate, MeshConfig, HarmonicPointForce, Sensor
from fem import FEMModel
from solvers import ModalFRFSolver, ModalSolver


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
        frequency_start=50.0,
        frequency_end=700.0,
        n_points=800,
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


def solve_case(use_black_hole: bool, n_modes: int = 30, zeta: float = 0.01):
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

    print("Calcul des modes...")
    modal_solver = ModalSolver(model)
    basis = modal_solver.solve_basis(n_modes=n_modes)

    print(f"Nombre de modes retenus : {basis.frequencies_hz.size}")
    print(f"Dernière fréquence modale retenue : {basis.frequencies_hz[-1]:.2f} Hz")

    print("Calcul FRF modal...")
    modal_frf_solver = ModalFRFSolver(model)
    result = modal_frf_solver.solve(
        excitation=excitation,
        sensor=sensor,
        n_modes=n_modes,
        damping_ratio=zeta,
        modal_basis=basis,
    )

    print(f"Noeud excitation : {result.excitation_node_id} | écart = {result.excitation_distance_m*1e3:.3f} mm")
    print(f"Noeud capteur    : {result.sensor_node_id} | écart = {result.sensor_distance_m*1e3:.3f} mm")

    return result, basis


def plot_frf(result_ref, result_bh) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(result_ref.frequencies_hz, result_ref.magnitude, label="Plaque uniforme")
    ax.semilogy(result_bh.frequencies_hz, result_bh.magnitude, label="Plaque avec VABH")
    ax.set_xlabel("Fréquence [Hz]")
    ax.set_ylabel("|FRF| [m/N]")
    ax.set_title("Comparaison des FRF modales")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(result_ref.frequencies_hz, result_ref.phase_deg, label="Plaque uniforme")
    ax.plot(result_bh.frequencies_hz, result_bh.phase_deg, label="Plaque avec VABH")
    ax.set_xlabel("Fréquence [Hz]")
    ax.set_ylabel("Phase [deg]")
    ax.set_title("Phase des FRF modales")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()


def plot_retained_modes(basis_ref, basis_bh) -> None:
    n_ref = np.arange(1, len(basis_ref.frequencies_hz) + 1)
    n_bh = np.arange(1, len(basis_bh.frequencies_hz) + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_ref, basis_ref.frequencies_hz, "o-", label="Uniforme")
    ax.plot(n_bh, basis_bh.frequencies_hz, "s-", label="VABH")
    ax.set_xlabel("Numéro de mode retenu")
    ax.set_ylabel("Fréquence [Hz]")
    ax.set_title("Modes retenus pour la reconstruction modale")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()


def main() -> None:
    n_modes = 30
    zeta = 0.01

    result_ref, basis_ref = solve_case(use_black_hole=False, n_modes=n_modes, zeta=zeta)
    result_bh, basis_bh = solve_case(use_black_hole=True, n_modes=n_modes, zeta=zeta)

    plot_frf(result_ref, result_bh)
    plot_retained_modes(basis_ref, basis_bh)
    plt.show()


if __name__ == "__main__":
    main()