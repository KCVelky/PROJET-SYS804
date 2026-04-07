from __future__ import annotations

import matplotlib.pyplot as plt

from solid3d import (
    ModalFRFSolver3D,
    HarmonicPointForce3D,
    PointSensor3D,
    Solid3DFEMModel,
    Solid3DMeshOptions,
)
from models import BlackHole, Material, Plate


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
        boundary_condition="clamped",
        black_hole=black_hole,
        name="Plaque 3D solide avec ABH",
    )
    plate.validate_black_hole_inside()
    return plate


def build_mesh_options() -> Solid3DMeshOptions:
    return Solid3DMeshOptions(
        element_order=2,
        global_size=0.025,
        local_size=0.010,
        local_refinement_radius=0.020,
        transition_thickness=0.010,
        top_surface_nu=11,
        top_surface_nv=9,
        algorithm_3d=10,
        save_msh_path="outputs/plate_3d_frf_modal.msh",
        optimize_high_order=False,
    )


def main() -> None:
    plate = build_plate()
    mesh_options = build_mesh_options()

    model = Solid3DFEMModel(
        plate=plate,
        mesh_options=mesh_options,
        use_black_hole=True,
        verbose=True,
    )
    model.build()

    excitation = HarmonicPointForce3D(
        x=0.10,
        y=0.10,
        z=0.002,
        amplitude=1.0,
        frequency_start=80.0,
        frequency_end=300.0,
        n_points=200,
        phase_deg=0.0,
        direction="z",
    )
    sensor = PointSensor3D(
        x=0.35,
        y=0.25,
        z=0.002,
        name="S1",
        direction="z",
        response_type="displacement",
    )

    solver = ModalFRFSolver3D(model, verbose=True)
    result = solver.solve(
        excitation=excitation,
        sensor=sensor,
        n_modes=30,
        damping_ratio=0.01,
    )

    print("=== FRF modale 3D ===")
    print("Noeud excitation :", result.excitation_node_id)
    print("Noeud capteur    :", result.sensor_node_id)
    print("DDL excitation   :", result.excitation_dof_id)
    print("DDL capteur      :", result.sensor_dof_id)
    print("Modes utilisés   :", result.n_modes_used)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(result.frequencies_hz, result.magnitude)
    ax.set_xlabel("Fréquence [Hz]")
    ax.set_ylabel("|H(ω)|")
    ax.set_title("FRF modale 3D")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()