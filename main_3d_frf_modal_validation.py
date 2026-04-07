from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from models import BlackHole, Material, Plate
from solvers.damping import RayleighDamping
from solid3d import (
    FRFSolver3D,
    HarmonicPointForce3D,
    ModalFRFSolver3D,
    PointSensor3D,
    Solid3DFEMModel,
    Solid3DMeshOptions,
)


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
        global_size=0.020,
        local_size=0.008,
        local_refinement_radius=0.020,
        transition_thickness=0.010,
        top_surface_nu=13,
        top_surface_nv=11,
        algorithm_3d=10,
        save_msh_path="outputs/plate_3d_frf_modal_validation.msh",
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
        n_points=25,   # garder modéré car FRF directe
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

    damping = RayleighDamping.from_modal_damping_ratio(
        zeta=0.01,
        freq1_hz=100.0,
        freq2_hz=250.0,
    )

    direct_solver = FRFSolver3D(model, verbose=True)
    direct_result = direct_solver.solve(
        excitation=excitation,
        sensor=sensor,
        damping=damping,
    )

    modal_solver = ModalFRFSolver3D(model, verbose=True)
    modal_result = modal_solver.solve(
        excitation=excitation,
        sensor=sensor,
        n_modes=40,
        damping=damping,
    )

    mag_direct = np.abs(direct_result.frf_complex)
    mag_modal = np.abs(modal_result.frf_complex)

    rel_err_mag = np.linalg.norm(mag_modal - mag_direct) / max(np.linalg.norm(mag_direct), 1e-30)
    rel_err_cplx = np.linalg.norm(modal_result.frf_complex - direct_result.frf_complex) / max(
        np.linalg.norm(direct_result.frf_complex), 1e-30
    )

    print("\n=== Validation FRF 3D : direct vs modal ===")
    print(f"Erreur relative magnitude : {rel_err_mag:.6e}")
    print(f"Erreur relative complexe  : {rel_err_cplx:.6e}")
    print(f"Modes utilisés            : {modal_result.n_modes_used}")

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].semilogy(direct_result.frequencies_hz, mag_direct, label="Directe", linewidth=2)
    axes[0].semilogy(modal_result.frequencies_hz, mag_modal, "--", label="Modale", linewidth=2)
    axes[0].set_ylabel("|H(ω)|")
    axes[0].set_title("Validation FRF 3D")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        direct_result.frequencies_hz,
        100.0 * np.abs(mag_modal - mag_direct) / np.maximum(mag_direct, 1e-30),
        linewidth=2,
    )
    axes[1].set_xlabel("Fréquence [Hz]")
    axes[1].set_ylabel("Erreur [%]")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()