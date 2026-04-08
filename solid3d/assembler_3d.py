from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags

from models.plate import Plate
from solid3d.mesh_data_3d import Mesh3DData
from solid3d.tet10_element import LinearElasticTet10Element


class Solid3DAssembler:
    """Assembleur global 3D solide optimisé pour Tetra10."""

    _PERM_SWAP_34 = np.array([0, 1, 3, 2, 4, 8, 7, 6, 5, 9], dtype=np.int64)

    def __init__(self, plate: Plate, mesh: Mesh3DData, verbose: bool = True) -> None:
        self.plate = plate
        self.mesh = mesh
        self.verbose = verbose

        if self.mesh.n_nodes_per_cell != 10:
            raise NotImplementedError(
                "Cette V1 du solveur 3D gère uniquement les tétraèdres quadratiques Tetra10."
            )

        self.element = LinearElasticTet10Element(
            young_modulus=plate.material.E,
            poisson_ratio=plate.material.nu,
            density=plate.material.rho,
        )

    @staticmethod
    def element_dof_map(node_ids: np.ndarray) -> np.ndarray:
        node_ids = np.asarray(node_ids, dtype=np.int64)
        return (3 * node_ids[:, None] + np.array([0, 1, 2], dtype=np.int64)).ravel()

    @staticmethod
    def _corner_signed_jacobian(node_coords: np.ndarray) -> float:
        x1, x2, x3, x4 = node_coords[:4]
        J = np.column_stack([x2 - x1, x3 - x1, x4 - x1])
        return float(np.linalg.det(J))

    def _canonicalize_tet10_connectivity(self, elem: np.ndarray) -> np.ndarray:
        coords = self.mesh.points[elem, :]
        det_corner = self._corner_signed_jacobian(coords)
        if det_corner > 0.0:
            return elem

        elem2 = elem[self._PERM_SWAP_34]
        coords2 = self.mesh.points[elem2, :]
        det_corner2 = self._corner_signed_jacobian(coords2)
        if det_corner2 > 0.0:
            return elem2

        raise ValueError(
            "Impossible de corriger l'orientation locale d'un tétra10. "
            f"det_corner initial = {det_corner:.6e}, det_corner après permutation = {det_corner2:.6e}"
        )

    def assemble(self) -> tuple[csr_matrix, csr_matrix]:
        n_nodes = self.mesh.n_points
        n_dofs = 3 * n_nodes
        n_elems = self.mesh.n_cells

        if self.verbose:
            print("=== Assemblage 3D ===")
            print("Nombre de noeuds   :", n_nodes)
            print("Nombre de ddl      :", n_dofs)
            print("Nombre d'éléments  :", n_elems)

        tri_r, tri_c = np.triu_indices(30)
        n_terms = tri_r.size
        total_terms = n_elems * n_terms

        I = np.empty(total_terms, dtype=np.int32)
        J = np.empty(total_terms, dtype=np.int32)
        VK = np.empty(total_terms, dtype=float)
        VM = np.empty(total_terms, dtype=float)

        progress_step = max(1, n_elems // 20)

        for e, elem_raw in enumerate(self.mesh.cells):
            elem = self._canonicalize_tet10_connectivity(elem_raw)
            coords = self.mesh.points[elem, :]

            try:
                Ke, Me = self.element.stiffness_and_mass(coords)
            except Exception as exc:
                det_corner = self._corner_signed_jacobian(coords)
                raise ValueError(
                    f"Échec sur l'élément {e} | det_corner = {det_corner:.6e} | "
                    f"connectivité locale = {elem.tolist()} | erreur : {exc}"
                ) from exc

            dofs = self.element_dof_map(elem)
            sl = slice(e * n_terms, (e + 1) * n_terms)
            I[sl] = dofs[tri_r]
            J[sl] = dofs[tri_c]
            VK[sl] = Ke[tri_r, tri_c]
            VM[sl] = Me[tri_r, tri_c]

            if self.verbose and ((e + 1) % progress_step == 0 or e == n_elems - 1):
                print(f"Assemblage : {e+1}/{n_elems} éléments")

        K_upper = coo_matrix((VK, (I, J)), shape=(n_dofs, n_dofs)).tocsr()
        M_upper = coo_matrix((VM, (I, J)), shape=(n_dofs, n_dofs)).tocsr()

        K = K_upper + K_upper.T - diags(K_upper.diagonal())
        M = M_upper + M_upper.T - diags(M_upper.diagonal())
        return K.tocsr(), M.tocsr()
