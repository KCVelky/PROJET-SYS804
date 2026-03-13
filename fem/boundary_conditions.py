# fem/boundary_conditions.py

from __future__ import annotations

import numpy as np

from models.plate import Plate
from fem.mesh_generator import MeshData


def find_boundary_nodes(plate: Plate, mesh: MeshData, tol: float = 1e-12) -> np.ndarray:
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]

    mask = (
        np.isclose(x, 0.0, atol=tol)
        | np.isclose(x, plate.Lx, atol=tol)
        | np.isclose(y, 0.0, atol=tol)
        | np.isclose(y, plate.Ly, atol=tol)
    )

    return np.where(mask)[0]


def constrained_dofs_from_bc(plate: Plate, mesh: MeshData) -> np.ndarray:
    """
    Convention :
    - simply_supported : w = 0 sur le bord
    - clamped          : w = 0, theta_x = 0, theta_y = 0 sur le bord
    - free             : aucun ddl imposé
    """
    bc = plate.boundary_condition
    boundary_nodes = find_boundary_nodes(plate, mesh)

    constrained = []

    if bc == "simply_supported":
        for nid in boundary_nodes:
            constrained.append(3 * nid + 0)

    elif bc == "clamped":
        for nid in boundary_nodes:
            constrained.extend([3 * nid + 0, 3 * nid + 1, 3 * nid + 2])

    elif bc == "free":
        constrained = []

    else:
        raise ValueError(f"Condition limite non gérée : {bc}")

    if len(constrained) == 0:
        return np.array([], dtype=int)

    return np.unique(np.array(constrained, dtype=int))


def free_dofs_from_constrained(n_dofs: int, constrained_dofs: np.ndarray) -> np.ndarray:
    all_dofs = np.arange(n_dofs, dtype=int)
    if constrained_dofs.size == 0:
        return all_dofs
    return np.setdiff1d(all_dofs, constrained_dofs)