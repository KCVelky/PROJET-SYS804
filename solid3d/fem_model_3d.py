# solid3d/fem_model_3d.py

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from models.plate import Plate
from solid3d.assembler_3d import Solid3DAssembler
from solid3d.boundary_conditions_3d import (
    constrained_dofs_3d,
    free_dofs_from_constrained_3d,
)
from solid3d.gmsh_mesher_3d import GmshABH3DMesher, Solid3DMeshOptions
from solid3d.mesh_data_3d import Mesh3DData


class Solid3DFEMModel:
    """
    Modèle EF 3D solide :
    - génération du maillage 3D
    - assemblage K et M
    - application des conditions aux limites
    """

    def __init__(
        self,
        plate: Plate,
        mesh_options: Solid3DMeshOptions,
        use_black_hole: bool = True,
        verbose: bool = True,
    ) -> None:
        self.plate = plate
        self.mesh_options = mesh_options
        self.use_black_hole = use_black_hole
        self.verbose = verbose

        self.mesh: Mesh3DData | None = None
        self.K: csr_matrix | None = None
        self.M: csr_matrix | None = None

        self.constrained_dofs: np.ndarray | None = None
        self.free_dofs: np.ndarray | None = None

        self.Kff: csr_matrix | None = None
        self.Mff: csr_matrix | None = None

    @property
    def n_nodes(self) -> int:
        return 0 if self.mesh is None else self.mesh.n_points

    @property
    def n_dofs(self) -> int:
        return 3 * self.n_nodes

    def build_mesh(self) -> None:
        if self.verbose:
            print("=== Génération du maillage 3D ===")

        mesher = GmshABH3DMesher(
            plate=self.plate,
            use_black_hole=self.use_black_hole,
            model_name="solid3d_modal_model",
        )
        self.mesh = mesher.generate(self.mesh_options)

        if self.verbose:
            print("Points maillage   :", self.mesh.n_points)
            print("Cellules maillage :", self.mesh.n_cells)
            print("Ordre élément     :", self.mesh.element_order)

    def assemble(self) -> None:
        if self.mesh is None:
            raise RuntimeError("Le maillage doit être généré avant l'assemblage.")

        assembler = Solid3DAssembler(
            plate=self.plate,
            mesh=self.mesh,
            verbose=self.verbose,
        )
        self.K, self.M = assembler.assemble()

    def apply_boundary_conditions(self) -> None:
        if self.mesh is None or self.K is None or self.M is None:
            raise RuntimeError("Le maillage et les matrices doivent exister avant les CL.")

        self.constrained_dofs = constrained_dofs_3d(self.plate, self.mesh)
        self.free_dofs = free_dofs_from_constrained_3d(self.n_dofs, self.constrained_dofs)

        self.Kff = self.K[self.free_dofs, :][:, self.free_dofs]
        self.Mff = self.M[self.free_dofs, :][:, self.free_dofs]

        if self.verbose:
            print("DDL totaux   :", self.n_dofs)
            print("DDL bloqués  :", len(self.constrained_dofs))
            print("DDL libres   :", len(self.free_dofs))

    def build(self) -> None:
        self.build_mesh()
        self.assemble()
        self.apply_boundary_conditions()

    def expand_reduced_vector(self, vec_free: np.ndarray) -> np.ndarray:
        if self.free_dofs is None:
            raise RuntimeError("Les ddl libres ne sont pas disponibles.")
        full = np.zeros(self.n_dofs, dtype=float)
        full[self.free_dofs] = vec_free
        return full

    def find_nearest_node(self, x: float, y: float, z: float) -> tuple[int, float]:
        if self.mesh is None:
            raise RuntimeError("Le maillage doit être construit avant la recherche de noeud.")

        target = np.array([x, y, z], dtype=float)
        distances = np.linalg.norm(self.mesh.points - target, axis=1)
        node_id = int(np.argmin(distances))
        return node_id, float(distances[node_id])