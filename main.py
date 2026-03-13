# main.py

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from models import Material, BlackHole, Plate, MeshConfig
from fem import FEMModel
from solvers import ModalSolver


def plot_mode(mesh_nodes: np.ndarray, mesh_elements: np.ndarray, mode_vector: np.ndarray, title: str) -> None:
    """
    Affiche la déformée transverse w d'un mode.
    """
    w = mode_vector[0::3]

    tri = mtri.Triangulation(
        mesh_nodes[:, 0] * 1e3,
        mesh_nodes[:, 1] * 1e3,
        triangles=mesh_elements
    )

    w_plot = w / np.max(np.abs(w))

    fig, ax = plt.subplots(figsize=(8, 6))
    contour = ax.tricontourf(tri, w_plot, levels=40)
    plt.colorbar(contour, ax=ax, label="Amplitude modale normalisée")

    ax.triplot(tri, color="k", linewidth=0.25, alpha=0.35)
    ax.set_title(title)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def main() -> None:
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

    mesh_config = MeshConfig(
        element_size=0.02,
        refine_near_black_hole=True,
        refinement_radius=0.08,
        refinement_element_size=0.005,
        element_type="tri3",
    )

    model = FEMModel(
        plate=plate,
        mesh_config=mesh_config,
        use_black_hole=True,
    )
    model.build()

    print("=== Modèle EF construit ===")
    print("Noeuds       :", model.n_nodes)
    print("DDL totaux   :", model.n_dofs)
    print("DDL bloqués  :", len(model.constrained_dofs))
    print("DDL libres   :", len(model.free_dofs))

    solver = ModalSolver(model)
    freqs, modes = solver.solve(n_modes=6)

    print("\n=== Premières fréquences propres [Hz] ===")
    for i, f in enumerate(freqs, start=1):
        print(f"Mode {i:02d} : {f:.3f} Hz")

    for i in range(min(4, modes.shape[1])):
        plot_mode(
            model.mesh.nodes,
            model.mesh.elements,
            modes[:, i],
            title=f"Mode {i+1} - f = {freqs[i]:.2f} Hz"
        )

    plt.show()


if __name__ == "__main__":
    main()