# main_3d_modal.py

from __future__ import annotations

from models import Material, BlackHole, Plate
from solid3d import (
    ModalSolver3D,
    Solid3DFEMModel,
    Solid3DMeshOptions,
    Solid3DMeshViewer,
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
        boundary_condition="clamped",   # recommandé pour cette V1 3D
        black_hole=black_hole,
        name="3D solid plate with top-machined ABH",
    )

    plate.validate_black_hole_inside()
    return plate


def main() -> None:
    plate = build_plate()

    # Commence volontairement plus grossier que ton preview 3D,
    # sinon l'assemblage/modal peut devenir très lourd.
    mesh_options = Solid3DMeshOptions(
        element_order=2,
        global_size=0.012,
        local_size=0.004,
        local_refinement_radius=0.030,
        transition_thickness=0.010,
        top_surface_nu=17,
        top_surface_nv=13,
        algorithm_3d=10,
        save_msh_path="outputs/plate_abh_3d_modal.msh",
        optimize_high_order=False,
    )

    model = Solid3DFEMModel(
        plate=plate,
        mesh_options=mesh_options,
        use_black_hole=True,
        verbose=True,
    )
    model.build()

    solver = ModalSolver3D(model, verbose=True)
    basis = solver.solve_basis(n_modes=6)

    print("\n=== Premières fréquences propres 3D [Hz] ===")
    for i, f in enumerate(basis.frequencies_hz, start=1):
        print(f"Mode {i:02d} : {f:.3f} Hz")

    mode_to_plot = 1
    Solid3DMeshViewer.show_mode_shape(
        mesh=model.mesh,
        mode_vector=basis.modes_full[:, mode_to_plot - 1],
        title=f"Mode {mode_to_plot} | f = {basis.frequencies_hz[mode_to_plot - 1]:.3f} Hz",
        normalize=True,
        scale=None,
        color_by="uz",
    )


if __name__ == "__main__":
    main()