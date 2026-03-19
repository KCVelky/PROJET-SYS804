# solid3d/mesh_data_3d.py

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh3DData:
    """
    Données du maillage volumique 3D.
    """

    points: np.ndarray                       # (N, 3)
    cells: np.ndarray                        # (M, n_nodes_per_cell)
    n_nodes_per_cell: int
    gmsh_element_type: int
    element_order: int
    node_tags_gmsh: np.ndarray
    element_tags_gmsh: np.ndarray | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return int(self.points.shape[0])

    @property
    def n_cells(self) -> int:
        return int(self.cells.shape[0])

    def to_pyvista(self):
        """
        Convertit le maillage en pyvista.UnstructuredGrid.
        """
        import pyvista as pv

        if self.n_nodes_per_cell == 4:
            vtk_cell_type = pv.CellType.TETRA
        elif self.n_nodes_per_cell == 10:
            if hasattr(pv.CellType, "QUADRATIC_TETRA"):
                vtk_cell_type = pv.CellType.QUADRATIC_TETRA
            else:
                raise RuntimeError(
                    "Cette version de PyVista/VTK ne semble pas exposer QUADRATIC_TETRA."
                )
        else:
            raise ValueError(
                f"Type de tétra non géré pour la visualisation : {self.n_nodes_per_cell} nœuds/élément."
            )

        n_cells = self.n_cells
        cells_flat = np.column_stack(
            [
                np.full(n_cells, self.n_nodes_per_cell, dtype=np.int64),
                self.cells.astype(np.int64),
            ]
        ).ravel()

        celltypes = np.full(n_cells, int(vtk_cell_type), dtype=np.uint8)
        grid = pv.UnstructuredGrid(cells_flat, celltypes, self.points.astype(float))

        # scalaires utiles
        grid.cell_data["cell_id"] = np.arange(self.n_cells, dtype=np.int32)
        grid.point_data["point_id"] = np.arange(self.n_points, dtype=np.int32)

        for key, value in self.metadata.items():
            if np.isscalar(value):
                grid.field_data[key] = np.array([value])

        return grid