# solid3d/boundary_conditions_3d.py

from __future__ import annotations

import numpy as np

from models.plate import Plate
from solid3d.mesh_data_3d import Mesh3DData


def find_perimeter_nodes_3d(
    plate: Plate,
    mesh: Mesh3DData,
    tol: float | None = None,
) -> np.ndarray:
    """
    Noeuds situés sur les faces latérales du pourtour :
    x = 0, x = Lx, y = 0, y = Ly
    """
    if tol is None:
        tol = 1e-9 * max(plate.Lx, plate.Ly, plate.h0)

    x = mesh.points[:, 0]
    y = mesh.points[:, 1]

    mask = (
        np.isclose(x, 0.0, atol=tol)
        | np.isclose(x, plate.Lx, atol=tol)
        | np.isclose(y, 0.0, atol=tol)
        | np.isclose(y, plate.Ly, atol=tol)
    )

    return np.where(mask)[0]


def constrained_dofs_3d(plate: Plate, mesh: Mesh3DData) -> np.ndarray:
    """
    Convention 3D solide V1 :
    - clamped : u = v = w = 0 sur les faces latérales du pourtour
    - free    : aucun ddl imposé
    - simply_supported : non implémenté proprement en 3D solide dans cette V1
    """
    bc = plate.boundary_condition

    if bc == "free":
        return np.array([], dtype=np.int64)

    if bc == "simply_supported":
        raise NotImplementedError(
            "La condition 'simply_supported' n'est pas encore implémentée proprement en 3D solide. "
            "Utilise 'clamped' ou 'free' pour cette branche 3D."
        )

    if bc != "clamped":
        raise ValueError(f"Condition limite non gérée en 3D : {bc}")

    boundary_nodes = find_perimeter_nodes_3d(plate, mesh)
    constrained = []

    for nid in boundary_nodes:
        constrained.extend([3 * nid + 0, 3 * nid + 1, 3 * nid + 2])

    return np.unique(np.array(constrained, dtype=np.int64))


def free_dofs_from_constrained_3d(n_dofs: int, constrained_dofs: np.ndarray) -> np.ndarray:
    all_dofs = np.arange(n_dofs, dtype=np.int64)
    if constrained_dofs.size == 0:
        return all_dofs
    return np.setdiff1d(all_dofs, constrained_dofs)