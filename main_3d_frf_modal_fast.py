from __future__ import annotations

import hashlib
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
)

RECOMPUTE_MESH = False
RECOMPUTE_BASIS = False
RECOMPUTE_FRF = False

N_MODES_USED = 60
N_MODES_CACHE = 100


def short_cache_name(prefix: str, payload: str, ext: str) -> str:
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}.{ext}"


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
        save_msh_path="outputs/plate_3d_frf_modal_fast.msh",
        reuse_saved_msh=not RECOMPUTE_MESH,
        optimize_high_order=False,
    )


def build_mesh_filename(
    plate: Plate,
    mesh_options: Solid3DMeshOptions,
    use_black_hole: bool,
) -> str:
    bh = plate.black_hole
    payload = [
        f"bc={plate.boundary_condition}",
        f"eo={mesh_options.element_order}",
        f"gs={mesh_options.global_size}",
        f"ls={mesh_options.local_size}",
        f"rr={mesh_options.local_refinement_radius}",
        f"tt={mesh_options.transition_thickness}",
        f"nu={mesh_options.top_surface_nu}",
        f"nv={mesh_options.top_surface_nv}",
        f"alg={mesh_options.algorithm_3d}",
        f"use_bh={use_black_hole}",
    ]

    if use_black_hole and bh is not None and bh.enabled:
        payload += [
            f"xc={bh.xc}",
            f"yc={bh.yc}",
            f"r={bh.radius}",
            f"rt={bh.truncation_radius}",
            f"hr={bh.residual_thickness}",
            f"m={bh.exponent}",
        ]

    return short_cache_name("mesh3d", "|".join(payload), "msh")


def build_basis_filename(
    plate: Plate,
    mesh_options: Solid3DMeshOptions,
    n_modes: int,
    use_black_hole: bool,
    mesh_filename: str,
) -> str:
    bh = plate.black_hole
    payload = [
        f"mesh={Path(mesh_filename).stem}",
        f"bc={plate.boundary_condition}",
        f"nm={n_modes}",
        f"use_bh={use_black_hole}",
        f"eo={mesh_options.element_order}",
    ]

    if use_black_hole and bh is not None and bh.enabled:
        payload += [
            f"r={bh.radius}",
            f"rt={bh.truncation_radius}",
            f"hr={bh.residual_thickness}",
            f"m={bh.exponent}",
        ]

    return short_cache_name("modal_basis_3d", "|".join(payload), "npz")


def build_dofs_filename(mesh_filename: str, plate: Plate) -> str:
    payload = f"mesh={Path(mesh_filename).stem}|bc={plate.boundary_condition}"
    return short_cache_name("dofs3d", payload, "npz")


def build_frf_filename(
    mesh_filename: str,
    plate: Plate,
    excitation: HarmonicPointForce3D,
    sensor: PointSensor3D,
    damping: RayleighDamping,
    n_modes_used: int,
) -> str:
    payload = [
        f"mesh={Path(mesh_filename).stem}",
        f"bc={plate.boundary_condition}",
        f"nm={n_modes_used}",
        f"fx={excitation.x}",
        f"fy={excitation.y}",
        f"fz={excitation.z}",
        f"fdir={excitation.direction}",
        f"amp={excitation.amplitude}",
        f"fstart={excitation.frequency_start}",
        f"fend={excitation.frequency_end}",
        f"np={excitation.n_points}",
        f"sx={sensor.x}",
        f"sy={sensor.y}",
        f"sz={sensor.z}",
        f"sdir={sensor.direction}",
        f"rtype={sensor.response_type}",
        f"ra={damping.alpha}",
        f"rb={damping.beta}",
    ]
    return short_cache_name("frf3d_modal", "|".join(payload), "npz")


def save_frf_result(filepath: str | Path, result) -> None:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
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

    n_modes = N_MODES_USED
    n_modes_cache = max(N_MODES_CACHE, n_modes)
    use_black_hole = True

    mesh_filename = build_mesh_filename(
        plate=plate,
        mesh_options=mesh_options,
        use_black_hole=use_black_hole,
    )
    mesh_options.save_msh_path = str(Path("outputs") / mesh_filename)

    model = Solid3DFEMModel(
        plate=plate,
        mesh_options=mesh_options,
        use_black_hole=use_black_hole,
        verbose=True,
    )

    excitation = HarmonicPointForce3D(
        x=0.10,
        y=0.10,
        z=0.002,
        amplitude=1.0,
        frequency_start=80.0,
        frequency_end=300.0,
        n_points=1000,
        phase_deg=0.0,
        direction="z",
    )

    sensor = PointSensor3D(
        x=0.35,
        y=0.25,
        z=0.002,
        name="S1",
        direction="z",
        response_type="displacement",
    )

    damping = RayleighDamping.from_modal_damping_ratio(
        zeta=0.01,
        freq1_hz=100.0,
        freq2_hz=250.0,
    )

    basis_filename = build_basis_filename(
        plate=plate,
        mesh_options=mesh_options,
        n_modes=n_modes_cache,
        use_black_hole=use_black_hole,
        mesh_filename=mesh_filename,
    )
    basis_path = Path("outputs") / basis_filename

    dofs_filename = build_dofs_filename(mesh_filename=mesh_filename, plate=plate)
    dofs_path = Path("outputs") / dofs_filename

    frf_filename = build_frf_filename(
        mesh_filename=mesh_filename,
        plate=plate,
        excitation=excitation,
        sensor=sensor,
        damping=damping,
        n_modes_used=n_modes,
    )
    frf_path = Path("outputs") / frf_filename

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

        if n_cached >= n_modes:
            modal_basis = solver.modal_solver.truncate_basis(cached_basis, n_modes)
        else:
            modal_basis = solver.modal_solver.solve_basis(n_modes=n_modes_cache)
            solver.modal_solver.save_basis(modal_basis, basis_path)
            modal_basis = solver.modal_solver.truncate_basis(modal_basis, n_modes)
    else:
        modal_basis = solver.modal_solver.solve_basis(n_modes=n_modes_cache)
        solver.modal_solver.save_basis(modal_basis, basis_path)
        modal_basis = solver.modal_solver.truncate_basis(modal_basis, n_modes)

    if frf_path.exists() and not RECOMPUTE_FRF:
        frequencies_hz, frf_complex = load_frf_arrays(frf_path)
    else:
        result = solver.solve(
            excitation=excitation,
            sensor=sensor,
            n_modes=n_modes,
            damping=damping,
            modal_basis=modal_basis,
        )
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