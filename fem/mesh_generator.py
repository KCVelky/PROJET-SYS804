# fem/mesh_generator.py

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.plate import Plate
from models.mesh_config import MeshConfig


@dataclass
class MeshData:
    """
    Structure de stockage du maillage.
    - nodes : tableau (N, 2)
    - elements : tableau (M, 3) d'indices de noeuds
    """
    nodes: np.ndarray
    elements: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray
    thickness_nodal: np.ndarray | None = None

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]


class StructuredTriMeshGenerator:
    """
    Générateur de maillage triangulaire structuré sur plaque rectangulaire.

    Idée :
    - grille cartésienne en x/y
    - raffinement local autour du trou noir
    - chaque maille quadrangulaire est découpée en 2 triangles
    """

    def __init__(self, plate: Plate, mesh_config: MeshConfig) -> None:
        self.plate = plate
        self.mesh_config = mesh_config

    @staticmethod
    def _march_axis(start: float, stop: float, step: float) -> np.ndarray:
        """
        Construit un axe [start, stop] avec pas approx. = step,
        en garantissant l'inclusion exacte de stop.
        """
        coords = [start]
        x = start

        while x + step < stop - 1e-14:
            x += step
            coords.append(x)

        if coords[-1] < stop:
            coords.append(stop)

        return np.array(coords, dtype=float)

    def _build_axis_with_optional_refinement(
        self,
        length: float,
        center: float | None,
        local_half_width: float | None,
    ) -> np.ndarray:
        """
        Construit un axe 1D éventuellement raffiné dans une zone locale.
        """
        coarse_h = self.mesh_config.element_size

        if (
            (not self.mesh_config.refine_near_black_hole)
            or (self.plate.black_hole is None)
            or (center is None)
            or (local_half_width is None)
        ):
            return self._march_axis(0.0, length, coarse_h)

        fine_h = self.mesh_config.refinement_element_size

        a = max(0.0, center - local_half_width)
        b = min(length, center + local_half_width)

        if a >= b:
            return self._march_axis(0.0, length, coarse_h)

        left = self._march_axis(0.0, a, coarse_h)
        middle = self._march_axis(a, b, fine_h)
        right = self._march_axis(b, length, coarse_h)

        coords = np.concatenate([left, middle, right])
        coords = np.unique(np.round(coords, decimals=12))
        coords.sort()

        if coords[0] != 0.0:
            coords = np.insert(coords, 0, 0.0)
        if coords[-1] != length:
            coords = np.append(coords, length)

        return coords

    def generate(
        self,
        use_black_hole_for_thickness: bool = True,
        use_black_hole_region_for_refinement: bool = True,
    ) -> MeshData:
        """
        Génère le maillage 2D triangulaire.

        - use_black_hole_for_thickness :
            active ou non le champ d'épaisseur variable

        - use_black_hole_region_for_refinement :
            utilise la zone du trou noir pour raffiner le maillage,
            même si l'épaisseur variable est désactivée
        """
        bh = self.plate.black_hole if use_black_hole_region_for_refinement else None

        if bh is not None and self.mesh_config.refine_near_black_hole:
            local_half_width_x = bh.radius + self.mesh_config.refinement_radius
            local_half_width_y = bh.radius + self.mesh_config.refinement_radius

            x_coords = self._build_axis_with_optional_refinement(
                self.plate.Lx, bh.xc, local_half_width_x
            )
            y_coords = self._build_axis_with_optional_refinement(
                self.plate.Ly, bh.yc, local_half_width_y
            )
        else:
            x_coords = self._march_axis(0.0, self.plate.Lx, self.mesh_config.element_size)
            y_coords = self._march_axis(0.0, self.plate.Ly, self.mesh_config.element_size)

        nx = len(x_coords)
        ny = len(y_coords)

        X, Y = np.meshgrid(x_coords, y_coords)
        nodes = np.column_stack([X.ravel(), Y.ravel()])

        def node_id(i: int, j: int) -> int:
            return j * nx + i

        elements = []

        for j in range(ny - 1):
            for i in range(nx - 1):
                n1 = node_id(i, j)
                n2 = node_id(i + 1, j)
                n3 = node_id(i, j + 1)
                n4 = node_id(i + 1, j + 1)

                if (i + j) % 2 == 0:
                    elements.append([n1, n2, n4])
                    elements.append([n1, n4, n3])
                else:
                    elements.append([n1, n2, n3])
                    elements.append([n2, n4, n3])

        elements = np.array(elements, dtype=int)

        thickness_nodal = np.array(
            [
                self.plate.thickness_at(x, y, use_black_hole=use_black_hole_for_thickness)
                for x, y in nodes
            ],
            dtype=float,
        )

        return MeshData(
            nodes=nodes,
            elements=elements,
            x_coords=x_coords,
            y_coords=y_coords,
            thickness_nodal=thickness_nodal,
        )   