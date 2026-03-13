# fem/assembler.py

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix

from models.plate import Plate
from fem.mesh_generator import MeshData
from fem.element_matrices import MindlinTri3Element


class GlobalAssembler:
    """
    Assembleur global des matrices K et M.
    """

    def __init__(self, plate: Plate, mesh: MeshData) -> None:
        self.plate = plate
        self.mesh = mesh

        self.element = MindlinTri3Element(
            young_modulus=plate.material.E,
            poisson_ratio=plate.material.nu,
            density=plate.material.rho,
        )

    @staticmethod
    def element_dof_map(node_ids: np.ndarray) -> np.ndarray:
        """
        3 ddl par noeud : [w, tx, ty]
        """
        dofs = []
        for nid in node_ids:
            dofs.extend([3 * nid + 0, 3 * nid + 1, 3 * nid + 2])
        return np.array(dofs, dtype=int)

    def assemble(self) -> tuple[csr_matrix, csr_matrix]:
        n_nodes = self.mesh.n_nodes
        n_dofs = 3 * n_nodes

        K = lil_matrix((n_dofs, n_dofs), dtype=float)
        M = lil_matrix((n_dofs, n_dofs), dtype=float)

        for elem in self.mesh.elements:
            coords = self.mesh.nodes[elem, :]
            h_elem = float(np.mean(self.mesh.thickness_nodal[elem]))

            Ke = self.element.stiffness_matrix(coords, h_elem)
            Me = self.element.mass_matrix(coords, h_elem)

            dofs = self.element_dof_map(elem)

            for a in range(9):
                A = dofs[a]
                for b in range(9):
                    B = dofs[b]
                    K[A, B] += Ke[a, b]
                    M[A, B] += Me[a, b]

        return K.tocsr(), M.tocsr()