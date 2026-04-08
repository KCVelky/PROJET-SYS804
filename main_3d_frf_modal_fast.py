from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models import BlackHole, Material, Plate
from solvers.damping import RayleighDamping
from solid3d import (
    HarmonicPointForce3D,
    ModalFRFSolver3D,
    PointSensor3D,
    Solid3DFEMModel,
    Solid3DMeshOptions,
    build_basis_cache_filename,
    build_dofs_cache_filename,
    build_frf_cache_filename,
    build_mesh_cache_filename,
)

RECOMPUTE_MESH = False
RECOMPUTE_BASIS = False
RECOMPUTE_FRF = False

N_MODES_USED = 60
N_MODES_CACHE = 100


def build_plate() -> Plate:
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
        boundary_condition="clamped",
        black_hole=black_hole,
        name="Plaque 3D solide avec ABH",
    )
    plate.validate_black_hole_inside()
    return plate


def build_mesh_options() -> Solid3DMeshOptions:
    return Solid3DMeshOptions(
        element_order=2,
        global_size=0.020,
        local_size=0.008,
        local_refinement_radius=0.020,
        transition_thickness=0.010,
        top_surface_nu=13,
        top_surface_nv=11,
        algorithm_3d=10,
        save_msh_path="outputs",
        reuse_saved_msh=not RECOMPUTE_MESH,
        optimize_high_order=False,
    )


def save_frf_result(filepath: str | Path, result) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        frequencies_hz=result.frequencies_hz,
        response_complex=result.response_complex,
        frf_complex=result.frf_complex,
        excitation_node_id=np.array([result.excitation_node_id], dtype=np.int64),
        sensor_node_id=np.array([result.sensor_node_id], dtype=np.int64),
        excitation_dof_id=np.array([result.excitation_dof_id], dtype=np.int64),
        sensor_dof_id=np.array([result.sensor_dof_id], dtype=np.int64),
        excitation_distance_m=np.array([result.excitation_distance_m], dtype=float),
        sensor_distance_m=np.array([result.sensor_distance_m], dtype=float),
        excitation_direction=np.array([result.excitation_direction]),
        sensor_direction=np.array([result.sensor_direction]),
        response_type=np.array([result.response_type]),
        n_modes_used=np.array([result.n_modes_used], dtype=np.int64),
        retained_mode_frequencies_hz=result.retained_mode_frequencies_hz,
    )


def load_frf_arrays(filepath: str | Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(filepath)
    return data["frequencies_hz"], data["frf_complex"]


def main() -> None:
    plate = build_plate()
    mesh_options = build_mesh_options()
    use_black_hole = True

    mesh_filename = build_mesh_cache_filename(plate=plate, mesh_options=mesh_options, use_black_hole=use_black_hole)
    mesh_options.save_msh_path = str(Path("outputs") / mesh_filename)

    model = Solid3DFEMModel(plate=plate, mesh_options=mesh_options, use_black_hole=use_black_hole, verbose=True)

    excitation = HarmonicPointForce3D(
        x=0.10, y=0.10, z=0.002, amplitude=1.0,
        frequency_start=80.0, frequency_end=300.0, n_points=1000,
        phase_deg=0.0, direction="z",
    )
    sensor = PointSensor3D(x=0.35, y=0.25, z=0.002, name="S1", direction="z", response_type="displacement")
    damping = RayleighDamping.from_modal_damping_ratio(zeta=0.01, freq1_hz=100.0, freq2_hz=250.0)

    basis_path = Path("outputs") / build_basis_cache_filename(
        plate=plate,
        mesh_filename=mesh_filename,
        n_modes=max(N_MODES_CACHE, N_MODES_USED),
        use_black_hole=use_black_hole,
        storage="compact",
    )
    dofs_path = Path("outputs") / build_dofs_cache_filename(mesh_filename=mesh_filename, plate=plate)
    frf_path = Path("outputs") / build_frf_cache_filename(
        plate=plate,
        mesh_filename=mesh_filename,
        excitation=excitation,
        sensor=sensor,
        damping=damping,
        n_modes_used=N_MODES_USED,
    )

    basis_cached = basis_path.exists() and not RECOMPUTE_BASIS
    dofs_cached = dofs_path.exists()

    if basis_cached and dofs_cached:
        model.build_mesh()
        model.load_dof_partition(dofs_path)
    elif basis_cached:
        model.build_mesh_and_dofs_only()
        model.save_dof_partition(dofs_path)
    else:
        model.build()
        model.save_dof_partition(dofs_path)

    solver = ModalFRFSolver3D(model, verbose=True)

    if basis_cached:
        cached_basis = solver.modal_solver.load_basis(basis_path)
        n_cached = cached_basis.modes_free.shape[1]
        if n_cached >= N_MODES_USED:
            modal_basis = solver.modal_solver.truncate_basis(cached_basis, N_MODES_USED)
        else:
            modal_basis = solver.modal_solver.solve_basis(n_modes=max(N_MODES_CACHE, N_MODES_USED), build_full_modes=False)
            solver.modal_solver.save_basis(modal_basis, basis_path, store_full_modes=False, compressed=False, dtype=np.float32)
            modal_basis = solver.modal_solver.truncate_basis(modal_basis, N_MODES_USED)
    else:
        modal_basis = solver.modal_solver.solve_basis(n_modes=max(N_MODES_CACHE, N_MODES_USED), build_full_modes=False)
        solver.modal_solver.save_basis(modal_basis, basis_path, store_full_modes=False, compressed=False, dtype=np.float32)
        modal_basis = solver.modal_solver.truncate_basis(modal_basis, N_MODES_USED)

    if frf_path.exists() and not RECOMPUTE_FRF:
        frequencies_hz, frf_complex = load_frf_arrays(frf_path)
    else:
        result = solver.solve(excitation=excitation, sensor=sensor, n_modes=N_MODES_USED, damping=damping, modal_basis=modal_basis)
        save_frf_result(frf_path, result)
        frequencies_hz = result.frequencies_hz
        frf_complex = result.frf_complex

    magnitude = np.abs(frf_complex)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogy(frequencies_hz, magnitude, linewidth=2)
    ax.set_xlabel("Fréquence [Hz]")
    ax.set_ylabel("|H(ω)|")
    ax.set_title("FRF modale 3D rapide")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
