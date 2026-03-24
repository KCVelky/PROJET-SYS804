# main_3d_mesh_preview.py

from __future__ import annotations

from models import Material, BlackHole, Plate
from solid3d import GmshABH3DMesher, Solid3DMeshOptions, Solid3DMeshViewer


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
        boundary_condition="clamped",   # en 3D, on part plus proprement là-dessus
        black_hole=black_hole,
        name="Plate 3D with top-machined ABH",
    )

    plate.validate_black_hole_inside()
    return plate


def main() -> None:
    plate = build_plate()

    mesher = GmshABH3DMesher(
        plate=plate,
        use_black_hole=True,
        model_name="plate_3d_abh",
    )

    options = Solid3DMeshOptions(
        element_order=2,            # tétraédrique quadratique
        global_size=0.020,          # 20 mm
        local_size=0.005,           # 5 mm
        local_refinement_radius=0.030,
        transition_thickness=0.020,
        top_surface_nu=25,
        top_surface_nv=21,
        algorithm_3d=10,            # HXT
        save_msh_path="outputs/plate_abh_3d.msh",
    )

    mesh = mesher.generate(options)

    print("=== Maillage 3D généré ===")
    print("Nombre de points         :", mesh.n_points)
    print("Nombre de cellules       :", mesh.n_cells)
    print("Nœuds par cellule        :", mesh.n_nodes_per_cell)
    print("Ordre élément            :", mesh.element_order)
    print("Type Gmsh                :", mesh.gmsh_element_type)
    print("Nom Gmsh                 :", mesh.metadata.get("gmsh_element_name"))

    Solid3DMeshViewer.show(mesh, title="Plaque 3D + VBH usiné par le dessus")


if __name__ == "__main__":
    main()