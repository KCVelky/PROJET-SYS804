from __future__ import annotations

from models import BlackHole, Material, Plate
from solid3d import ModalSolver3D, ModalValidation3D, Solid3DFEMModel, Solid3DMeshOptions


def build_plate() -> Plate:
    return Plate(
        length_x=0.50,
        length_y=0.40,
        thickness=0.002,
        material=Material.aluminum(),
        boundary_condition="clamped",
        black_hole=BlackHole(
            xc=0.25,
            yc=0.20,
            radius=0.06,
            truncation_radius=0.005,
            residual_thickness=0.0003,
            exponent=2.0,
            enabled=True,
        ),
        name="Plaque 3D solide avec ABH",
    )


def build_mesh_options() -> Solid3DMeshOptions:
    return Solid3DMeshOptions(
        element_order=2,
        global_size=0.012,
        local_size=0.004,
        local_refinement_radius=0.030,
        transition_thickness=0.010,
        top_surface_nu=17,
        top_surface_nv=13,
        algorithm_3d=10,
        save_msh_path="outputs/plate_abh_3d_modal_validation.msh",
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

    solver = ModalSolver3D(model, verbose=True)
    basis = solver.solve_basis(n_modes=6)

    validator = ModalValidation3D(model, verbose=True)
    result = validator.validate(n_modes=6, modal_basis=basis)

    print("\nValidation OK ?", validator.is_valid(result))


if __name__ == "__main__":
    main()
