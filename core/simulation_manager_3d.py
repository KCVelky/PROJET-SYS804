from __future__ import annotations

from typing import Any

from models import BlackHole, Material, Plate
from solvers import RayleighDamping
from solid3d import (
    FRFSolver3D,
    HarmonicPointForce3D,
    ModalComparison3D,
    ModalSolver3D,
    ModalValidation3D,
    PointSensor3D,
    Solid3DFEMModel,
    Solid3DMeshOptions,
)


class SimulationManager3D:
    """
    Chef d'orchestre dédié à la branche 3D solide.

    Cette classe reste séparée du SimulationManager 2D afin de :
    - ne pas casser l'existant,
    - garder une architecture 3D propre,
    - préparer l'intégration Qt et la suite FRF/modal FRF.
    """

    def __init__(self, load_defaults: bool = True) -> None:
        self.material_params: dict[str, Any] = {}
        self.plate_params: dict[str, Any] = {}
        self.black_hole_params: dict[str, Any] = {}
        self.mesh_params: dict[str, Any] = {}
        self.excitation_params: dict[str, Any] = {}
        self.sensors: dict[str, dict[str, Any]] = {}

        self._model_cache: dict[bool, Solid3DFEMModel] = {}
        self._modal_basis_cache: dict[tuple[bool, int], Any] = {}
        self.last_results: dict[str, Any] = {}

        if load_defaults:
            self.load_default_case()

    def invalidate_cache(self) -> None:
        self._model_cache.clear()
        self._modal_basis_cache.clear()

    def reset(self) -> None:
        self.material_params.clear()
        self.plate_params.clear()
        self.black_hole_params.clear()
        self.mesh_params.clear()
        self.excitation_params.clear()
        self.sensors.clear()
        self.last_results.clear()
        self.invalidate_cache()

    def load_default_case(self) -> None:
        self.set_material(
            young_modulus=69e9,
            poisson_ratio=0.33,
            density=2700.0,
            name="Aluminum",
        )
        self.set_plate(
            length_x=0.50,
            length_y=0.40,
            thickness=0.002,
            boundary_condition="clamped",
            name="Plaque 3D solide",
        )
        self.set_black_hole(
            xc=0.25,
            yc=0.20,
            radius=0.06,
            truncation_radius=0.005,
            residual_thickness=0.0003,
            exponent=2.0,
            enabled=True,
        )
        self.set_mesh(
            element_order=2,
            global_size=0.012,
            local_size=0.004,
            local_refinement_radius=0.030,
            transition_thickness=0.010,
            top_surface_nu=17,
            top_surface_nv=13,
            algorithm_3d=10,
            save_msh_path="outputs/plate_abh_3d_manager.msh",
            optimize_high_order=False,
        )
        self.set_excitation(
            x=0.10,
            y=0.10,
            z=0.002,
            amplitude=1.0,
            frequency_start=50.0,
            frequency_end=700.0,
            n_points=300,
            phase_deg=0.0,
            direction="z",
        )
        self.set_sensor(
            name="S1",
            x=0.35,
            y=0.25,
            z=0.002,
            direction="z",
            response_type="displacement",
        )

    def set_material(
        self,
        young_modulus: float,
        poisson_ratio: float,
        density: float,
        name: str = "Generic material",
    ) -> None:
        self.material_params = {
            "young_modulus": young_modulus,
            "poisson_ratio": poisson_ratio,
            "density": density,
            "name": name,
        }
        self.invalidate_cache()

    def set_plate(
        self,
        length_x: float,
        length_y: float,
        thickness: float,
        boundary_condition: str = "clamped",
        name: str = "Plaque 3D solide",
    ) -> None:
        self.plate_params = {
            "length_x": length_x,
            "length_y": length_y,
            "thickness": thickness,
            "boundary_condition": boundary_condition,
            "name": name,
        }
        self.invalidate_cache()

    def set_black_hole(
        self,
        xc: float,
        yc: float,
        radius: float,
        truncation_radius: float,
        residual_thickness: float,
        exponent: float = 2.0,
        enabled: bool = True,
    ) -> None:
        self.black_hole_params = {
            "xc": xc,
            "yc": yc,
            "radius": radius,
            "truncation_radius": truncation_radius,
            "residual_thickness": residual_thickness,
            "exponent": exponent,
            "enabled": enabled,
        }
        self.invalidate_cache()

    def set_mesh(
        self,
        element_order: int = 2,
        global_size: float = 0.012,
        local_size: float = 0.004,
        local_refinement_radius: float = 0.030,
        transition_thickness: float = 0.010,
        top_surface_nu: int = 17,
        top_surface_nv: int = 13,
        algorithm_3d: int = 10,
        save_msh_path: str | None = "outputs/plate_abh_3d_manager.msh",
        optimize_high_order: bool = False,
        high_order_opt_mode: int = 2,
        high_order_num_layers: int = 6,
        high_order_pass_max: int = 25,
        high_order_threshold_min: float = 0.1,
        high_order_threshold_max: float = 2.0,
        high_order_fix_boundary_nodes: int = 0,
        high_order_prim_surf_mesh: int = 1,
        high_order_iter_max: int = 100,
    ) -> None:
        self.mesh_params = {
            "element_order": element_order,
            "global_size": global_size,
            "local_size": local_size,
            "local_refinement_radius": local_refinement_radius,
            "transition_thickness": transition_thickness,
            "top_surface_nu": top_surface_nu,
            "top_surface_nv": top_surface_nv,
            "algorithm_3d": algorithm_3d,
            "save_msh_path": save_msh_path,
            "optimize_high_order": optimize_high_order,
            "high_order_opt_mode": high_order_opt_mode,
            "high_order_num_layers": high_order_num_layers,
            "high_order_pass_max": high_order_pass_max,
            "high_order_threshold_min": high_order_threshold_min,
            "high_order_threshold_max": high_order_threshold_max,
            "high_order_fix_boundary_nodes": high_order_fix_boundary_nodes,
            "high_order_prim_surf_mesh": high_order_prim_surf_mesh,
            "high_order_iter_max": high_order_iter_max,
        }
        self.invalidate_cache()

    def set_excitation(
        self,
        x: float,
        y: float,
        z: float,
        amplitude: float,
        frequency_start: float,
        frequency_end: float,
        n_points: int = 300,
        phase_deg: float = 0.0,
        direction: str = "z",
    ) -> None:
        self.excitation_params = {
            "x": x,
            "y": y,
            "z": z,
            "amplitude": amplitude,
            "frequency_start": frequency_start,
            "frequency_end": frequency_end,
            "n_points": n_points,
            "phase_deg": phase_deg,
            "direction": direction,
        }

    def set_sensor(
        self,
        name: str,
        x: float,
        y: float,
        z: float,
        direction: str = "z",
        response_type: str = "displacement",
    ) -> None:
        self.sensors[name] = {
            "x": x,
            "y": y,
            "z": z,
            "name": name,
            "direction": direction,
            "response_type": response_type,
        }

    def enable_black_hole(self) -> None:
        if self.black_hole_params:
            self.black_hole_params["enabled"] = True
            self.invalidate_cache()

    def disable_black_hole(self) -> None:
        if self.black_hole_params:
            self.black_hole_params["enabled"] = False
            self.invalidate_cache()

    def build_material(self) -> Material:
        if not self.material_params:
            raise RuntimeError("Les paramètres matériau 3D ne sont pas définis.")
        return Material(**self.material_params)

    def build_black_hole(self) -> BlackHole | None:
        if not self.black_hole_params:
            return None
        return BlackHole(**self.black_hole_params)

    def build_plate(self) -> Plate:
        if not self.plate_params:
            raise RuntimeError("Les paramètres plaque 3D ne sont pas définis.")
        plate = Plate(
            length_x=self.plate_params["length_x"],
            length_y=self.plate_params["length_y"],
            thickness=self.plate_params["thickness"],
            material=self.build_material(),
            boundary_condition=self.plate_params["boundary_condition"],
            black_hole=self.build_black_hole(),
            name=self.plate_params["name"],
        )
        plate.validate_black_hole_inside()
        return plate

    def build_mesh_options(self) -> Solid3DMeshOptions:
        if not self.mesh_params:
            raise RuntimeError("Les paramètres de maillage 3D ne sont pas définis.")
        return Solid3DMeshOptions(**self.mesh_params)

    def build_excitation(self) -> HarmonicPointForce3D:
        if not self.excitation_params:
            raise RuntimeError("Les paramètres d'excitation 3D ne sont pas définis.")
        return HarmonicPointForce3D(**self.excitation_params)

    def build_sensor(self, name: str) -> PointSensor3D:
        if name not in self.sensors:
            raise KeyError(f"Capteur 3D inconnu : '{name}'")
        return PointSensor3D(**self.sensors[name])

    def get_model(self, use_black_hole: bool = True, rebuild: bool = False) -> Solid3DFEMModel:
        key = bool(use_black_hole)
        if (not rebuild) and key in self._model_cache:
            return self._model_cache[key]

        model = Solid3DFEMModel(
            plate=self.build_plate(),
            mesh_options=self.build_mesh_options(),
            use_black_hole=use_black_hole,
            verbose=True,
        )
        model.build()
        self._model_cache[key] = model
        return model

    def get_modal_basis(self, n_modes: int = 6, use_black_hole: bool = True, rebuild: bool = False):
        key = (bool(use_black_hole), int(n_modes))
        if (not rebuild) and key in self._modal_basis_cache:
            return self._modal_basis_cache[key]

        model = self.get_model(use_black_hole=use_black_hole, rebuild=rebuild)
        solver = ModalSolver3D(model, verbose=True)
        basis = solver.solve_basis(n_modes=n_modes)
        self._modal_basis_cache[key] = basis
        return basis

    def get_mesh_preview(self, use_black_hole: bool = True, rebuild: bool = False):
        return self.get_model(use_black_hole=use_black_hole, rebuild=rebuild).mesh

    def run_modal_analysis(
        self,
        n_modes: int = 6,
        use_black_hole: bool = True,
        rebuild: bool = False,
    ) -> dict[str, Any]:
        model = self.get_model(use_black_hole=use_black_hole, rebuild=rebuild)
        basis = self.get_modal_basis(n_modes=n_modes, use_black_hole=use_black_hole, rebuild=rebuild)
        result = {
            "model": model,
            "frequencies_hz": basis.frequencies_hz,
            "modes_full": basis.modes_full,
            "modes_free": basis.modes_free,
            "omegas_rad_s": basis.omegas_rad_s,
            "eigenvalues": basis.eigenvalues,
            "modal_masses": basis.modal_masses,
        }
        self.last_results["modal_3d"] = result
        return result

    def validate_modal_analysis(
        self,
        n_modes: int = 6,
        use_black_hole: bool = True,
        rebuild: bool = False,
    ) -> dict[str, Any]:
        model = self.get_model(use_black_hole=use_black_hole, rebuild=rebuild)
        basis = self.get_modal_basis(n_modes=n_modes, use_black_hole=use_black_hole, rebuild=rebuild)
        validator = ModalValidation3D(model, verbose=True)
        validation = validator.validate(n_modes=n_modes, modal_basis=basis)
        result = {
            "model": model,
            "basis": basis,
            "validation": validation,
            "is_valid": validator.is_valid(validation, stiffness_rel_tol=1e-2),
        }
        self.last_results["modal_validation_3d"] = result
        return result

    def compare_modal(self, n_modes: int = 6) -> dict[str, Any]:
        comparator = ModalComparison3D(
            plate=self.build_plate(),
            mesh_options=self.build_mesh_options(),
            verbose=True,
        )
        comparison = comparator.run(n_modes=n_modes)
        result = {
            "comparison": comparison,
            "frequencies_uniform_hz": comparison.frequencies_uniform_hz,
            "frequencies_abh_hz": comparison.frequencies_abh_hz,
        }
        self.last_results["modal_comparison_3d"] = result
        return result

    def run_frf_direct(
        self,
        sensor_name: str,
        use_black_hole: bool = True,
        damping_ratio: float = 0.01,
        damping_freq1_hz: float = 140.0,
        damping_freq2_hz: float = 500.0,
        rebuild: bool = False,
    ) -> dict[str, Any]:
        model = self.get_model(use_black_hole=use_black_hole, rebuild=rebuild)
        excitation = self.build_excitation()
        sensor = self.build_sensor(sensor_name)

        damping = RayleighDamping.from_modal_damping_ratio(
            zeta=damping_ratio,
            freq1_hz=damping_freq1_hz,
            freq2_hz=damping_freq2_hz,
        )

        solver = FRFSolver3D(model, verbose=True)
        frf_result = solver.solve(
            excitation=excitation,
            sensor=sensor,
            damping=damping,
        )

        result = {
            "model": model,
            "sensor": sensor,
            "excitation": excitation,
            "damping": damping,
            "frf_result": frf_result,
        }
        self.last_results["frf_direct_3d"] = result
        return result

    def compare_frf_direct(
        self,
        sensor_name: str,
        damping_ratio: float = 0.01,
        damping_freq1_hz: float = 140.0,
        damping_freq2_hz: float = 500.0,
        rebuild: bool = False,
    ) -> dict[str, Any]:
        ref = self.run_frf_direct(
            sensor_name=sensor_name,
            use_black_hole=False,
            damping_ratio=damping_ratio,
            damping_freq1_hz=damping_freq1_hz,
            damping_freq2_hz=damping_freq2_hz,
            rebuild=rebuild,
        )
        bh = self.run_frf_direct(
            sensor_name=sensor_name,
            use_black_hole=True,
            damping_ratio=damping_ratio,
            damping_freq1_hz=damping_freq1_hz,
            damping_freq2_hz=damping_freq2_hz,
            rebuild=rebuild,
        )
        result = {
            "reference": ref,
            "black_hole": bh,
        }
        self.last_results["frf_direct_comparison_3d"] = result
        return result

    def get_case_summary(self) -> dict[str, Any]:
        return {
            "material": dict(self.material_params),
            "plate": dict(self.plate_params),
            "black_hole": dict(self.black_hole_params),
            "mesh": dict(self.mesh_params),
            "excitation": dict(self.excitation_params),
            "sensors": {name: dict(params) for name, params in self.sensors.items()},
        }
