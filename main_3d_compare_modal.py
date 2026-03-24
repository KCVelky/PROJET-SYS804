from __future__ import annotations

import matplotlib.pyplot as plt

from models import BlackHole, Material, Plate
from solid3d import ModalComparison3D, Solid3DMeshOptions


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
        name="Plaque 3D solide avec ou sans ABH",
    )
    plate.validate_black_hole_inside()
    return plate


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
        save_msh_path="outputs/plate_3d_compare.msh",
        optimize_high_order=False,
    )


def main() -> None:
    plate = build_plate()
    mesh_options = build_mesh_options()

    comparator = ModalComparison3D(
        plate=plate,
        mesh_options=mesh_options,
        verbose=True,
    )
    result = comparator.run(n_modes=6)

    comparator.print_summary(result)
    comparator.plot_frequency_comparison(result)
    comparator.plot_relative_shift(result)
    comparator.plot_localization_comparison(result)

    # comparator.show_mode_pair(result, mode_number=1, color_by="umag")
    plt.show()


if __name__ == "__main__":
    main()
