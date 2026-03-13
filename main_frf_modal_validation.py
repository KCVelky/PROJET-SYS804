# main_frf_modal_validation.py

from __future__ import annotations

import matplotlib.pyplot as plt

from models import Material, BlackHole, Plate, MeshConfig, HarmonicPointForce, Sensor
from fem import FEMModel
from solvers import FRFSolver, ModalFRFSolver, RayleighDamping, ModalSolver


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
        name="Plate with VABH",
    )

    plate.validate_black_hole_inside()
    return plate


def main() -> None:
    plate = build_plate()

    mesh_config = MeshConfig(
        element_size=0.02,
        refine_near_black_hole=True,
        refinement_radius=0.08,
        refinement_element_size=0.005,
        element_type="tri3",
    )

    excitation = HarmonicPointForce(
        x=0.10,
        y=0.10,
        amplitude=1.0,
        frequency_start=80.0,
        frequency_end=600.0,
        n_points=80,
        phase_deg=0.0,
        direction="w",
    )

    sensor = Sensor(
        x=0.35,
        y=0.25,
        name="S1",
        response_type="displacement",
    )

    model = FEMModel(
        plate=plate,
        mesh_config=mesh_config,
        use_black_hole=True,
        refine_with_black_hole_region=True,
    )
    model.build()

    # direct
    damping_direct = RayleighDamping.from_modal_damping_ratio(
        zeta=0.01,
        freq1_hz=140.0,
        freq2_hz=500.0,
    )

    frf_direct_solver = FRFSolver(model)
    result_direct = frf_direct_solver.solve(
        excitation=excitation,
        sensor=sensor,
        damping=damping_direct,
    )

    # modal
    modal_solver = ModalSolver(model)
    basis = modal_solver.solve_basis(n_modes=30)

    frf_modal_solver = ModalFRFSolver(model)
    result_modal = frf_modal_solver.solve(
        excitation=excitation,
        sensor=sensor,
        n_modes=30,
        damping_ratio=0.01,
        modal_basis=basis,
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(result_direct.frequencies_hz, result_direct.magnitude, "o-", label="Direct")
    ax.semilogy(result_modal.frequencies_hz, result_modal.magnitude, "s--", label="Modal")
    ax.set_xlabel("Fréquence [Hz]")
    ax.set_ylabel("|FRF| [m/N]")
    ax.set_title("Validation : FRF directe vs FRF modale")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()