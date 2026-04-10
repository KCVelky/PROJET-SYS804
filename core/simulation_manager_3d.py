from __future__ import annotations

from typing import Any

import numpy as np

from models import BlackHole, Material, Plate
from solvers import RayleighDamping
from solid3d import (
    FRFSolver3D,
    HarmonicPointForce3D,
    ModalComparison3D,
    ModalSolver3D,
    ModalFRFSolver3D,
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
    
    # ============================================================
    # Bridge UI 3D <-> backend existant
    # ============================================================

    def _ui_use_black_hole(self, params: dict[str, Any]) -> bool:
        return str(params.get("case", "vbh")).lower() != "uniform"

    def _apply_ui_params(self, params: dict[str, Any]) -> None:
        """
        Traduit les paramètres venant de l'UI 3D vers l'API backend existante.
        """
        plate = params.get("plate", {})
        material = params.get("material", {})
        vbh = params.get("vbh", {})
        mesh = params.get("mesh", {})
        frf = params.get("frf", {})

        use_black_hole = self._ui_use_black_hole(params)

        # -------------------------
        # Material
        # -------------------------
        self.set_material(
            young_modulus=float(material.get("E", 69e9)),
            poisson_ratio=float(material.get("nu", 0.33)),
            density=float(material.get("rho", 2700.0)),
            name="UI Material",
        )

        # -------------------------
        # Plate
        # -------------------------
        h0 = float(plate.get("h0", 0.002))
        self.set_plate(
            length_x=float(plate.get("lx", 0.50)),
            length_y=float(plate.get("ly", 0.40)),
            thickness=h0,
            boundary_condition="clamped",
            name="Plaque 3D UI",
        )

        # -------------------------
        # Black hole
        # -------------------------
        radius = float(vbh.get("radius", 0.06))
        truncation_radius = float(
            vbh.get(
                "truncation_radius",
                max(1e-4, min(0.005, 0.25 * radius))
            )
        )

        self.set_black_hole(
            xc=float(vbh.get("cx", 0.25)),
            yc=float(vbh.get("cy", 0.20)),
            radius=radius,
            truncation_radius=truncation_radius,
            residual_thickness=float(vbh.get("h_residual", 0.0003)),
            exponent=float(vbh.get("m", 2.0)),
            enabled=use_black_hole,
        )

        # -------------------------
        # Mesh
        # -------------------------
        h_global = float(mesh.get("h_global", 0.012))
        h_local = float(mesh.get("h_local", 0.004))
        refine_vbh = bool(mesh.get("refine_vbh", True))

        local_ref_radius = 0.60 * radius if refine_vbh else max(radius, h_global)

        self.set_mesh(
            element_order=int(mesh.get("order", 2)),
            global_size=h_global,
            local_size=h_local,
            local_refinement_radius=max(1e-4, local_ref_radius),
            transition_thickness=max(1e-4, 0.20 * radius),
            top_surface_nu=17,
            top_surface_nv=13,
            algorithm_3d=10,
            save_msh_path="outputs/ui_3d_generated.msh",
            optimize_high_order=False,
        )

        # -------------------------
        # Excitation
        # -------------------------
        excitation = frf.get("excitation", [0.10, 0.10, h0])
        response = frf.get("response", [0.30, 0.20, h0])

        ex = float(excitation[0]) if len(excitation) > 0 else 0.10
        ey = float(excitation[1]) if len(excitation) > 1 else 0.10
        ez = float(excitation[2]) if len(excitation) > 2 else h0

        rx = float(response[0]) if len(response) > 0 else 0.30
        ry = float(response[1]) if len(response) > 1 else 0.20
        rz = float(response[2]) if len(response) > 2 else h0

        fmin = float(frf.get("fmin", 1.0))
        fmax = float(frf.get("fmax", 1000.0))
        n_freq = int(frf.get("n_freq", 400))
        direction = str(frf.get("direction", "uz")).lower().replace("u", "")

        self.set_excitation(
            x=ex,
            y=ey,
            z=ez,
            amplitude=1.0,
            frequency_start=fmin,
            frequency_end=fmax,
            n_points=n_freq,
            phase_deg=0.0,
            direction=direction,
        )

        self.set_sensor(
            name="UI_SENSOR",
            x=rx,
            y=ry,
            z=rz,
            direction=direction,
            response_type="displacement",
        )

    def _build_mesh_stats(
        self,
        model: Solid3DFEMModel,
        mesh: Any,
        use_black_hole: bool,
    ) -> dict[str, Any]:
        stats: dict[str, Any] = {
            "use_black_hole": use_black_hole,
            "element_order": self.mesh_params.get("element_order"),
            "global_size": self.mesh_params.get("global_size"),
            "local_size": self.mesh_params.get("local_size"),
        }

        if hasattr(mesh, "n_points"):
            stats["n_points"] = int(mesh.n_points)
        if hasattr(mesh, "n_cells"):
            stats["n_cells"] = int(mesh.n_cells)
        if hasattr(mesh, "bounds"):
            stats["bounds"] = tuple(float(v) for v in mesh.bounds)

        for attr_name in [
            "n_dofs_total",
            "n_total_dofs",
            "n_dofs",
            "n_free_dofs",
            "n_blocked_dofs",
        ]:
            if hasattr(model, attr_name):
                value = getattr(model, attr_name)
                if isinstance(value, (int, float)):
                    stats[attr_name] = int(value)

        return stats

    def preview_mesh(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Méthode attendue par l'UI 3D.
        Prépare et retourne le maillage sans logique supplémentaire côté UI.
        """
        self._apply_ui_params(params)
        use_black_hole = self._ui_use_black_hole(params)

        model = self.get_model(use_black_hole=use_black_hole, rebuild=True)
        mesh = model.mesh

        return {
            "model": model,
            "mesh": mesh,
            "grid": mesh,
            "stats": self._build_mesh_stats(model, mesh, use_black_hole),
        }

    def generate_mesh(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Même logique que preview_mesh, mais nom explicite pour le bouton UI.
        """
        self._apply_ui_params(params)
        use_black_hole = self._ui_use_black_hole(params)

        model = self.get_model(use_black_hole=use_black_hole, rebuild=True)
        mesh = model.mesh

        return {
            "model": model,
            "mesh": mesh,
            "grid": mesh,
            "stats": self._build_mesh_stats(model, mesh, use_black_hole),
        }

    def solve_modal(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Méthode attendue par l'UI 3D.
        """
        self._apply_ui_params(params)
        use_black_hole = self._ui_use_black_hole(params)

        modal_params = params.get("modal", {})
        n_modes = int(modal_params.get("n_modes", 6))

        raw = self.run_modal_analysis(
            n_modes=n_modes,
            use_black_hole=use_black_hole,
            rebuild=True,
        )

        freqs = raw.get("frequencies_hz", [])
        if hasattr(freqs, "tolist"):
            freqs = freqs.tolist()

        return {
            "model": raw["model"],
            "modal_result": raw,
            "freqs": freqs,
            "frequencies": freqs,
            "modes_full": raw.get("modes_full"),
            "modes_free": raw.get("modes_free"),
            "eigenvalues": raw.get("eigenvalues"),
        }

    def get_mode_shape(
        self,
        modal_result: dict[str, Any],
        mode_index: int,
        component: str = "norm",
        scale: float = 1.0,
    ) -> dict[str, Any]:
        """
        Retourne un mesh déformé + champ scalaire pour affichage PyVista.
        """
        raw = modal_result.get("modal_result", modal_result)
        model = raw["model"]
        mesh = model.mesh

        if not hasattr(mesh, "copy"):
            raise RuntimeError("Le mesh 3D ne supporte pas la copie pour l'affichage modal.")

        grid = mesh.copy(deep=True)

        modes_full = raw.get("modes_full", None)
        if modes_full is None:
            raise RuntimeError("Aucun mode complet disponible dans le résultat modal.")

        vec = np.asarray(modes_full[:, mode_index]).reshape(-1)

        n_points = getattr(grid, "n_points", None)
        if n_points is None:
            raise RuntimeError("Impossible de déterminer le nombre de nœuds du mesh.")

        if vec.size < 3 * n_points:
            # Fallback si le vecteur n'est pas structuré comme 3 ddl par nœud
            scalars = np.abs(vec[:n_points])
            grid.point_data["mode_shape"] = scalars
            return {
                "mesh": grid,
                "grid": grid,
                "scalars_name": "mode_shape",
            }

        disp = vec[: 3 * n_points].reshape(n_points, 3)

        comp = str(component).lower()
        if comp == "ux":
            scalars = disp[:, 0]
        elif comp == "uy":
            scalars = disp[:, 1]
        elif comp == "uz":
            scalars = disp[:, 2]
        else:
            scalars = np.linalg.norm(disp, axis=1)

        if hasattr(grid, "points"):
            grid.points = np.asarray(grid.points).copy() + float(scale) * disp

        grid.point_data["mode_shape"] = scalars
        grid.point_data["ux"] = disp[:, 0]
        grid.point_data["uy"] = disp[:, 1]
        grid.point_data["uz"] = disp[:, 2]
        grid.point_data["unorm"] = np.linalg.norm(disp, axis=1)

        return {
            "mesh": grid,
            "grid": grid,
            "scalars_name": "mode_shape",
        }

    def compute_frf(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Bridge FRF pour l'UI.
        Pour l'instant, si l'UI demande une FRF modale, on retombe proprement
        sur la FRF directe tant que la version modale UI-bridge n'est pas branchée.
        """
        self._apply_ui_params(params)
        use_black_hole = self._ui_use_black_hole(params)

        frf_params = params.get("frf", {})
        method_requested = str(frf_params.get("method", "directe")).lower()

        fmin = float(frf_params.get("fmin", 1.0))
        fmax = float(frf_params.get("fmax", 1000.0))

        eta = float(self.material_params.get("density", 2700.0))  # placeholder pour éviter un zéro impossible
        _ = eta  # juste pour expliciter qu'on ne s'en sert pas directement ici

        result = self.run_frf_direct(
            sensor_name="UI_SENSOR",
            use_black_hole=use_black_hole,
            damping_ratio=0.01,
            damping_freq1_hz=max(1.0, fmin),
            damping_freq2_hz=max(fmin + 1.0, min(max(fmax, fmin + 1.0), 500.0)),
            rebuild=True,
        )

        frf_result = result["frf_result"]

        freq = None
        H = None

        if isinstance(frf_result, dict):
            freq = frf_result.get("frequencies_hz", None) or frf_result.get("freq_hz", None)
            H = frf_result.get("response", None) or frf_result.get("frf", None)
        else:
            if hasattr(frf_result, "frequencies_hz"):
                freq = frf_result.frequencies_hz
            elif hasattr(frf_result, "freq_hz"):
                freq = frf_result.freq_hz

            if hasattr(frf_result, "response"):
                H = frf_result.response
            elif hasattr(frf_result, "frf"):
                H = frf_result.frf

        if freq is None:
            freq = []
        if H is None:
            H = []

        return {
            "method_requested": method_requested,
            "method_used": "directe" if method_requested.startswith("mod") else method_requested,
            "freq": np.asarray(freq).reshape(-1),
            "frequencies": np.asarray(freq).reshape(-1),
            "H": np.asarray(H).reshape(-1),
            "frf_result": result,
        }

    def get_case_summary(self) -> dict[str, Any]:
        return {
            "material": dict(self.material_params),
            "plate": dict(self.plate_params),
            "black_hole": dict(self.black_hole_params),
            "mesh": dict(self.mesh_params),
            "excitation": dict(self.excitation_params),
            "sensors": {name: dict(params) for name, params in self.sensors.items()},
        }


    # ------------------------------------------------------------------
    # Pont UI <-> backend 3D
    # ------------------------------------------------------------------
    @staticmethod
    def _ui_dir_to_solver(direction: str | None, default: str = "z") -> str:
        if direction is None:
            return default
        d = str(direction).strip().lower()
        mapping = {"ux": "x", "uy": "y", "uz": "z", "x": "x", "y": "y", "z": "z"}
        return mapping.get(d, default)

    def _apply_ui_params(self, params: dict[str, Any]) -> None:
        """
        Traduit les paramètres issus de l'UI 3D vers l'API historique du manager.
        """
        if not params:
            return

        case_name = str(params.get("case", "vbh")).lower()
        use_black_hole = case_name != "uniform"

        plate = params.get("plate", {})
        material = params.get("material", {})
        vbh = params.get("vbh", {})
        mesh = params.get("mesh", {})
        modal = params.get("modal", {})
        frf = params.get("frf", {})

        # --- matériau ---
        self.set_material(
            young_modulus=float(material.get("E", self.material_params.get("young_modulus", 69e9))),
            poisson_ratio=float(material.get("nu", self.material_params.get("poisson_ratio", 0.33))),
            density=float(material.get("rho", self.material_params.get("density", 2700.0))),
            name=str(material.get("name", self.material_params.get("name", "Aluminum"))),
        )

        # --- plaque ---
        self.set_plate(
            length_x=float(plate.get("lx", plate.get("length_x", self.plate_params.get("length_x", 0.50)))),
            length_y=float(plate.get("ly", plate.get("length_y", self.plate_params.get("length_y", 0.40)))),
            thickness=float(plate.get("h0", plate.get("thickness", self.plate_params.get("thickness", 0.002)))),
            boundary_condition=str(plate.get("boundary_condition", self.plate_params.get("boundary_condition", "clamped"))),
            name=str(plate.get("name", self.plate_params.get("name", "Plaque 3D solide"))),
        )

        # --- VBH ---
        radius = float(vbh.get("radius", self.black_hole_params.get("radius", 0.06)))
        trunc = float(vbh.get("truncation_radius", self.black_hole_params.get("truncation_radius", max(0.1 * radius, 1e-4))))
        residual = float(vbh.get("h_residual", vbh.get("residual_thickness", self.black_hole_params.get("residual_thickness", 3e-4))))

        self.set_black_hole(
            xc=float(vbh.get("cx", vbh.get("xc", self.black_hole_params.get("xc", 0.25)))),
            yc=float(vbh.get("cy", vbh.get("yc", self.black_hole_params.get("yc", 0.20)))),
            radius=radius,
            truncation_radius=trunc,
            residual_thickness=residual,
            exponent=float(vbh.get("m", vbh.get("exponent", self.black_hole_params.get("exponent", 2.0)))),
            enabled=bool(vbh.get("enabled", use_black_hole)),
        )

        # --- maillage ---
        refine_vbh = bool(mesh.get("refine_vbh", True))
        local_ref_radius = radius if refine_vbh else 0.0
        base_name = "abh" if use_black_hole else "uniform"
        save_path = mesh.get("save_msh_path") or f"outputs/plate_abh_3d_manager_{base_name}.msh"

        self.set_mesh(
            element_order=int(mesh.get("order", self.mesh_params.get("element_order", 2))),
            global_size=float(mesh.get("h_global", self.mesh_params.get("global_size", 0.012))),
            local_size=float(mesh.get("h_local", self.mesh_params.get("local_size", 0.004))),
            local_refinement_radius=float(mesh.get("local_refinement_radius", local_ref_radius)),
            transition_thickness=float(mesh.get("transition_thickness", self.mesh_params.get("transition_thickness", max(radius * 0.25, 0.01)))),
            top_surface_nu=int(mesh.get("top_surface_nu", self.mesh_params.get("top_surface_nu", 17))),
            top_surface_nv=int(mesh.get("top_surface_nv", self.mesh_params.get("top_surface_nv", 13))),
            algorithm_3d=int(mesh.get("algorithm_3d", self.mesh_params.get("algorithm_3d", 10))),
            save_msh_path=str(save_path),
            optimize_high_order=bool(mesh.get("optimize_high_order", self.mesh_params.get("optimize_high_order", False))),
            high_order_opt_mode=int(mesh.get("high_order_opt_mode", self.mesh_params.get("high_order_opt_mode", 2))),
            high_order_num_layers=int(mesh.get("high_order_num_layers", self.mesh_params.get("high_order_num_layers", 6))),
            high_order_pass_max=int(mesh.get("high_order_pass_max", self.mesh_params.get("high_order_pass_max", 25))),
            high_order_threshold_min=float(mesh.get("high_order_threshold_min", self.mesh_params.get("high_order_threshold_min", 0.1))),
            high_order_threshold_max=float(mesh.get("high_order_threshold_max", self.mesh_params.get("high_order_threshold_max", 2.0))),
            high_order_fix_boundary_nodes=int(mesh.get("high_order_fix_boundary_nodes", self.mesh_params.get("high_order_fix_boundary_nodes", 0))),
            high_order_prim_surf_mesh=int(mesh.get("high_order_prim_surf_mesh", self.mesh_params.get("high_order_prim_surf_mesh", 1))),
            high_order_iter_max=int(mesh.get("high_order_iter_max", self.mesh_params.get("high_order_iter_max", 100))),
        )

        # --- excitation / capteur UI ---
        thickness = float(plate.get("h0", self.plate_params.get("thickness", 0.002)))
        ex_xyz = list(frf.get("excitation", [0.10, 0.10, thickness]))
        rp_xyz = list(frf.get("response", [0.30, 0.20, thickness]))
        while len(ex_xyz) < 3:
            ex_xyz.append(thickness)
        while len(rp_xyz) < 3:
            rp_xyz.append(thickness)

        dir_solver = self._ui_dir_to_solver(frf.get("direction", "uz"))
        fmin = float(frf.get("fmin", self.excitation_params.get("frequency_start", 50.0)))
        fmax = float(frf.get("fmax", self.excitation_params.get("frequency_end", 700.0)))
        n_points = int(frf.get("n_freq", self.excitation_params.get("n_points", 300)))

        self.set_excitation(
            x=float(ex_xyz[0]),
            y=float(ex_xyz[1]),
            z=float(ex_xyz[2] if ex_xyz[2] > 0.0 else thickness),
            amplitude=float(frf.get("amplitude", self.excitation_params.get("amplitude", 1.0))),
            frequency_start=fmin,
            frequency_end=fmax,
            n_points=n_points,
            phase_deg=float(frf.get("phase_deg", self.excitation_params.get("phase_deg", 0.0))),
            direction=dir_solver,
        )

        self.set_sensor(
            name="UI_SENSOR",
            x=float(rp_xyz[0]),
            y=float(rp_xyz[1]),
            z=float(rp_xyz[2] if rp_xyz[2] > 0.0 else thickness),
            direction=dir_solver,
            response_type=str(frf.get("response_type", "displacement")),
        )

        # stocker quelques infos UI utiles
        self.last_results["ui_context"] = {
            "use_black_hole": use_black_hole,
            "n_modes": int(modal.get("n_modes", 20)),
            "frf_method": str(frf.get("method", "directe")).lower(),
            "frf_n_modes": int(frf.get("n_modes", 40)),
            "display_component": str(modal.get("component", "norm")),
            "display_scale": float(modal.get("scale", 1.0)),
            "compare_uniform_vs_vbh": bool(modal.get("compare_uniform_vs_vbh", False)),
        }

    def _mesh_stats(self, model: Solid3DFEMModel) -> dict[str, Any]:
        mesh = model.mesh
        if mesh is None:
            return {}
        return {
            "n_points": int(mesh.n_points),
            "n_cells": int(mesh.n_cells),
            "element_order": int(mesh.element_order),
            "n_nodes_per_cell": int(mesh.n_nodes_per_cell),
            "boundary_condition": str(model.plate.boundary_condition),
            "use_black_hole": bool(model.use_black_hole),
        }

    def preview_mesh(self, params: dict[str, Any]) -> dict[str, Any]:
        self._apply_ui_params(params)
        use_black_hole = bool(self.last_results.get("ui_context", {}).get("use_black_hole", True))
        model = self.get_model(use_black_hole=use_black_hole, rebuild=True)

        grid = None
        if model.mesh is not None:
            try:
                grid = model.mesh.to_pyvista()
            except Exception:
                grid = None

        result = {
            "model": model,
            "mesh_data": model.mesh,
            "mesh": grid,
            "stats": self._mesh_stats(model),
        }
        self.last_results["mesh_preview_3d"] = result
        return result

    def generate_mesh(self, params: dict[str, Any]) -> dict[str, Any]:
        # pour cette branche, générer le maillage revient à construire le modèle 3D complet
        return self.preview_mesh(params)

    def solve_modal(self, params: dict[str, Any]) -> dict[str, Any]:
        self._apply_ui_params(params)
        ui = self.last_results.get("ui_context", {})
        use_black_hole = bool(ui.get("use_black_hole", True))
        n_modes = int(ui.get("n_modes", 20))

        analysis = self.run_modal_analysis(
            n_modes=n_modes,
            use_black_hole=use_black_hole,
            rebuild=False,
        )

        result = {
            "model": analysis["model"],
            "freqs": analysis["frequencies_hz"],
            "frequencies": analysis["frequencies_hz"],
            "modes_full": analysis["modes_full"],
            "modes_free": analysis["modes_free"],
            "omegas_rad_s": analysis["omegas_rad_s"],
            "eigenvalues": analysis["eigenvalues"],
            "modal_masses": analysis["modal_masses"],
        }

        if bool(ui.get("compare_uniform_vs_vbh", False)):
            try:
                comp = self.compare_modal(n_modes=n_modes)
                result["comparison"] = {
                    "frequencies_uniform_hz": comp["frequencies_uniform_hz"],
                    "frequencies_abh_hz": comp["frequencies_abh_hz"],
                }
            except Exception as exc:
                result["comparison_error"] = str(exc)

        self.last_results["modal_3d_ui"] = result
        return result

    def get_mode_shape(
        self,
        modal_result: dict[str, Any],
        mode_index: int,
        component: str = "norm",
        scale: float = 1.0,
    ) -> dict[str, Any]:
        model: Solid3DFEMModel = modal_result["model"]
        mesh = model.mesh
        if mesh is None:
            raise RuntimeError("Le maillage 3D n'est pas disponible pour l'affichage du mode.")

        modes_full = np.asarray(modal_result["modes_full"])
        if modes_full.ndim != 2:
            raise ValueError("modes_full doit être une matrice 2D.")
        if mode_index < 0 or mode_index >= modes_full.shape[1]:
            raise IndexError("Indice de mode hors limites.")

        disp = modes_full[:, mode_index].reshape(-1, 3).astype(float)
        max_amp = float(np.max(np.linalg.norm(disp, axis=1)))
        if max_amp > 1e-16:
            disp = disp / max_amp

        try:
            grid = mesh.to_pyvista()
        except Exception as exc:
            raise RuntimeError(f"Conversion PyVista impossible : {exc}") from exc

        grid.point_data["disp"] = disp
        grid.point_data["ux"] = disp[:, 0]
        grid.point_data["uy"] = disp[:, 1]
        grid.point_data["uz"] = disp[:, 2]
        grid.point_data["umag"] = np.linalg.norm(disp, axis=1)

        scalar_name = {"ux": "ux", "uy": "uy", "uz": "uz", "norm": "umag", "umag": "umag"}.get(str(component).lower(), "umag")
        warped = grid.warp_by_vector("disp", factor=float(scale))

        return {
            "mesh": warped,
            "grid": warped,
            "scalars_name": scalar_name,
            "mode_index": int(mode_index),
        }

    def compute_frf(self, params: dict[str, Any]) -> dict[str, Any]:
        self._apply_ui_params(params)
        ui = self.last_results.get("ui_context", {})
        use_black_hole = bool(ui.get("use_black_hole", True))
        method = str(ui.get("frf_method", "directe")).lower()
        sensor_name = "UI_SENSOR"

        # on utilise le facteur de pertes UI comme approx. simple du taux modal
        eta_base = float(params.get("material", {}).get("eta", 0.01))
        damping_ratio = max(0.0, float(eta_base))

        visco = params.get("visco", {})
        if use_black_hole and bool(visco.get("enabled", False)):
            vbh = params.get("vbh", {})
            plate = params.get("plate", {})

            r_patch = float(visco.get("radius", 0.0))
            h_patch = float(visco.get("thickness", 0.0))
            eta_patch = float(visco.get("eta", 0.15))

            r_vbh = max(1e-9, float(vbh.get("radius", 0.06)))
            h0 = max(1e-9, float(plate.get("h0", 0.002)))

            area_ratio = min(1.0, (r_patch / r_vbh) ** 2)
            thickness_ratio = min(1.0, h_patch / h0)

            concentration_gain = 4.0
            delta_zeta = concentration_gain * eta_patch * area_ratio * thickness_ratio

            damping_ratio = min(0.25, damping_ratio + delta_zeta)

        if method.startswith("modal"):
            n_modes = int(ui.get("frf_n_modes", 40))
            model = self.get_model(use_black_hole=use_black_hole, rebuild=False)
            basis = self.get_modal_basis(n_modes=n_modes, use_black_hole=use_black_hole, rebuild=False)
            solver = ModalFRFSolver3D(model, verbose=True)
            frf = solver.solve(
                excitation=self.build_excitation(),
                sensor=self.build_sensor(sensor_name),
                n_modes=n_modes,
                damping=damping_ratio if damping_ratio > 0.0 else 0.01,
                modal_basis=basis,
            )
            result = {
                "method": "modal",
                "freq": frf.frequencies_hz,
                "frequencies": frf.frequencies_hz,
                "H": frf.frf_complex,
                "frf": frf.frf_complex,
                "response": frf.response_complex,
                "frf_result": frf,
                "n_modes_used": int(frf.n_modes_used),
            }
        else:
            exc = self.build_excitation()
            f1 = max(1.0, float(exc.frequency_start))
            f2 = max(f1 + 1.0, min(float(exc.frequency_end), max(float(exc.frequency_end), 2.0 * f1)))
            direct = self.run_frf_direct(
                sensor_name=sensor_name,
                use_black_hole=use_black_hole,
                damping_ratio=damping_ratio if damping_ratio > 0.0 else 0.01,
                damping_freq1_hz=f1,
                damping_freq2_hz=f2,
                rebuild=False,
            )
            frf = direct["frf_result"]
            result = {
                "method": "direct",
                "freq": frf.frequencies_hz,
                "frequencies": frf.frequencies_hz,
                "H": frf.frf_complex,
                "frf": frf.frf_complex,
                "response": frf.response_complex,
                "frf_result": frf,
            }

        self.last_results["frf_3d_ui"] = result
        return result
