# solid3d/pyvista_viewer_3d.py

from __future__ import annotations

import numpy as np

from solid3d.mesh_data_3d import Mesh3DData


class Solid3DMeshViewer:
    """
    Viewer 3D du maillage volumique et des modes 3D.
    """

    @staticmethod
    def _extract_surface(grid, mesh: Mesh3DData):
        return grid.extract_surface(
            nonlinear_subdivision=1 if mesh.element_order > 1 else 0,
            algorithm="dataset_surface",
        )

    @staticmethod
    def show(mesh: Mesh3DData, title: str = "Maillage 3D VBH") -> None:
        import pyvista as pv

        grid = mesh.to_pyvista()

        plotter = pv.Plotter(title=title)
        plotter.set_background("white")

        plotter.add_mesh(
            grid,
            show_edges=True,
            opacity=0.18,
            label="Maillage volumique",
        )

        surface = Solid3DMeshViewer._extract_surface(grid, mesh)
        plotter.add_mesh(
            surface,
            show_edges=True,
            opacity=0.35,
            label="Surface extérieure",
        )

        plotter.add_axes()
        plotter.show_grid()
        plotter.add_title(
            f"{title} | points = {mesh.n_points} | cellules = {mesh.n_cells}",
            font_size=10,
        )
        plotter.show()

    @staticmethod
    def show_mode_shape(
        mesh: Mesh3DData,
        mode_vector: np.ndarray,
        title: str = "Mode 3D",
        normalize: bool = True,
        scale: float | None = None,
        color_by: str = "uz",
    ) -> None:
        import pyvista as pv

        if mode_vector.ndim != 1:
            raise ValueError("mode_vector doit être un vecteur 1D.")
        if mode_vector.shape[0] != 3 * mesh.n_points:
            raise ValueError("La taille de mode_vector ne correspond pas au nombre de noeuds du maillage.")

        disp = mode_vector.reshape(-1, 3).copy()

        if normalize:
            max_amp = np.max(np.linalg.norm(disp, axis=1))
            if max_amp > 1e-16:
                disp /= max_amp

        if scale is None:
            bbox = mesh.points.max(axis=0) - mesh.points.min(axis=0)
            scale = 0.08 * float(np.max(bbox))

        grid = mesh.to_pyvista()
        grid.point_data["disp"] = disp
        grid.point_data["ux"] = disp[:, 0]
        grid.point_data["uy"] = disp[:, 1]
        grid.point_data["uz"] = disp[:, 2]
        grid.point_data["umag"] = np.linalg.norm(disp, axis=1)

        undeformed_surface = Solid3DMeshViewer._extract_surface(grid, mesh)

        warped = grid.warp_by_vector("disp", factor=scale)
        warped_surface = Solid3DMeshViewer._extract_surface(warped, mesh)

        scalar_name = color_by if color_by in ("ux", "uy", "uz", "umag") else "uz"

        plotter = pv.Plotter(title=title)
        plotter.set_background("white")

        plotter.add_mesh(
            undeformed_surface,
            color="lightgray",
            style="wireframe",
            opacity=0.35,
            line_width=1.0,
            label="Non déformé",
        )

        plotter.add_mesh(
            warped_surface,
            scalars=scalar_name,
            show_edges=False,
            opacity=1.0,
            label="Déformé",
        )

        plotter.add_axes()
        plotter.show_grid()
        plotter.add_title(title, font_size=10)
        plotter.show()