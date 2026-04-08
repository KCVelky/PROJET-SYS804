from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from models.plate import Plate
from solvers.damping import RayleighDamping
from solid3d.gmsh_mesher_3d import Solid3DMeshOptions
from solid3d.probes_3d import HarmonicPointForce3D, PointSensor3D


def _f(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def short_cache_name(prefix: str, payload: str, ext: str) -> str:
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{digest}.{ext}"


def build_mesh_cache_filename(
    plate: Plate,
    mesh_options: Solid3DMeshOptions,
    use_black_hole: bool,
) -> str:
    bh = plate.black_hole
    payload = [
        f"Lx={_f(plate.Lx)}",
        f"Ly={_f(plate.Ly)}",
        f"h0={_f(plate.h0)}",
        f"bc={plate.boundary_condition}",
        f"eo={mesh_options.element_order}",
        f"gs={_f(mesh_options.global_size)}",
        f"ls={_f(mesh_options.local_size)}",
        f"rr={_f(mesh_options.local_refinement_radius)}",
        f"tt={_f(mesh_options.transition_thickness)}",
        f"nu={mesh_options.top_surface_nu}",
        f"nv={mesh_options.top_surface_nv}",
        f"alg={mesh_options.algorithm_3d}",
        f"use_bh={int(bool(use_black_hole))}",
    ]
    if use_black_hole and bh is not None and bh.enabled:
        payload += [
            f"xc={_f(bh.xc)}",
            f"yc={_f(bh.yc)}",
            f"r={_f(bh.radius)}",
            f"rt={_f(bh.truncation_radius)}",
            f"hr={_f(bh.residual_thickness)}",
            f"m={_f(bh.exponent)}",
        ]
    return short_cache_name("mesh3d", "|".join(payload), "msh")


def build_dofs_cache_filename(mesh_filename: str, plate: Plate) -> str:
    payload = f"mesh={Path(mesh_filename).stem}|bc={plate.boundary_condition}|ndof=3"
    return short_cache_name("dofs3d", payload, "npz")


def build_basis_cache_filename(
    plate: Plate,
    mesh_filename: str,
    n_modes: int,
    use_black_hole: bool,
    storage: str = "compact",
) -> str:
    bh = plate.black_hole
    payload = [
        f"mesh={Path(mesh_filename).stem}",
        f"bc={plate.boundary_condition}",
        f"nm={int(n_modes)}",
        f"use_bh={int(bool(use_black_hole))}",
        f"storage={storage}",
    ]
    if use_black_hole and bh is not None and bh.enabled:
        payload += [
            f"r={_f(bh.radius)}",
            f"rt={_f(bh.truncation_radius)}",
            f"hr={_f(bh.residual_thickness)}",
            f"m={_f(bh.exponent)}",
        ]
    return short_cache_name("modal_basis_3d", "|".join(payload), "npz")


def build_frf_cache_filename(
    plate: Plate,
    mesh_filename: str,
    excitation: HarmonicPointForce3D,
    sensor: PointSensor3D,
    damping: RayleighDamping,
    n_modes_used: int,
) -> str:
    payload = [
        f"mesh={Path(mesh_filename).stem}",
        f"bc={plate.boundary_condition}",
        f"nm={int(n_modes_used)}",
        f"fx={_f(excitation.x)}",
        f"fy={_f(excitation.y)}",
        f"fz={_f(excitation.z)}",
        f"fdir={excitation.direction}",
        f"amp={_f(excitation.amplitude)}",
        f"fstart={_f(excitation.frequency_start)}",
        f"fend={_f(excitation.frequency_end)}",
        f"np={int(excitation.n_points)}",
        f"sx={_f(sensor.x)}",
        f"sy={_f(sensor.y)}",
        f"sz={_f(sensor.z)}",
        f"sdir={sensor.direction}",
        f"rtype={sensor.response_type}",
        f"ra={_f(damping.alpha)}",
        f"rb={_f(damping.beta)}",
    ]
    return short_cache_name("frf3d_modal", "|".join(payload), "npz")
