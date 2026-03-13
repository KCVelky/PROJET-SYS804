# fem/fem_model.py

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix

from models.plate import Plate
from models.mesh_config import MeshConfig
from fem.mesh_generator import MeshData, StructuredTriMeshGenerator
from fem.assembler import GlobalAssembler
from fem.boundary_conditions import constrained_dofs_from_bc, free_dofs_from_constrained


class FEMModel:
    """
    Modèle EF global :
    - génère le maillage
    - assemble K et M
    - applique les conditions aux limites
    """

    def __init__(
        self,
        plate: Plate,
        mesh_config: MeshConfig,
        use_black_hole: bool = True,
        refine_with_black_hole_region: bool = True,
    ) -> None:
        self.plate = plate
        self.mesh_config = mesh_config
        self.use_black_hole = use_black_hole
        self.refine_with_black_hole_region = refine_with_black_hole_region

        self.mesh: MeshData | None = None
        self.K: csr_matrix | None = None
        self.M: csr_matrix | None = None

        self.constrained_dofs: np.ndarray | None = None
        self.free_dofs: np.ndarray | None = None

        self.Kff: csr_matrix | None = None
        self.Mff: csr_matrix | None = None

    def find_nearest_node(self, x: float, y: float) -> tuple[int, float]:
        """
        Retourne l'indice du noeud le plus proche et la distance associée.
        """
        if self.mesh is None:
            raise RuntimeError("Le maillage doit être construit avant la recherche de noeud.")

        target = np.array([x, y], dtype=float)
        distances = np.linalg.norm(self.mesh.nodes - target, axis=1)
        node_id = int(np.argmin(distances))
        return node_id, float(distances[node_id])

    @staticmethod
    def dof_index(node_id: int, dof_name: str = "w") -> int:
        """
        Convention :
        - w       -> 0
        - theta_x -> 1
        - theta_y -> 2
        """
        mapping = {
            "w": 0,
            "theta_x": 1,
            "theta_y": 2,
        }

        if dof_name not in mapping:
            raise ValueError(f"DDL inconnu : {dof_name}")

        return 3 * node_id + mapping[dof_name]

    def is_free_dof(self, dof_id: int) -> bool:
        if self.free_dofs is None:
            raise RuntimeError("Les conditions aux limites doivent être appliquées avant ce test.")
        pos = np.searchsorted(self.free_dofs, dof_id)
        return pos < len(self.free_dofs) and self.free_dofs[pos] == dof_id

    @property
    def n_nodes(self) -> int:
        if self.mesh is None:
            return 0
        return self.mesh.n_nodes

    @property
    def n_dofs(self) -> int:
        return 3 * self.n_nodes

    def build_mesh(self) -> None:
        generator = StructuredTriMeshGenerator(self.plate, self.mesh_config)
        self.mesh = generator.generate(
            use_black_hole_for_thickness=self.use_black_hole,
            use_black_hole_region_for_refinement=self.refine_with_black_hole_region,
        )

    def assemble(self) -> None:
        if self.mesh is None:
            raise RuntimeError("Le maillage doit être généré avant l'assemblage.")

        assembler = GlobalAssembler(self.plate, self.mesh)
        self.K, self.M = assembler.assemble()

    def apply_boundary_conditions(self) -> None:
        if self.mesh is None or self.K is None or self.M is None:
            raise RuntimeError("Le maillage et les matrices doivent exister avant les CL.")

        self.constrained_dofs = constrained_dofs_from_bc(self.plate, self.mesh)
        self.free_dofs = free_dofs_from_constrained(self.n_dofs, self.constrained_dofs)

        self.Kff = self.K[self.free_dofs, :][:, self.free_dofs]
        self.Mff = self.M[self.free_dofs, :][:, self.free_dofs]

    def build(self) -> None:
        self.build_mesh()
        self.assemble()
        self.apply_boundary_conditions()

    def expand_reduced_vector(self, vec_free: np.ndarray) -> np.ndarray:
        """
        Recompose un vecteur global complet en remettant 0 sur les ddl bloqués.
        """
        if self.free_dofs is None:
            raise RuntimeError("Les degrés de liberté libres ne sont pas disponibles.")

        full = np.zeros(self.n_dofs, dtype=float)
        full[self.free_dofs] = vec_free
        return full