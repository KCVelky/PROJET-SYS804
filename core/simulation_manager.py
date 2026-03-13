# core/simulation_manager.py

from __future__ import annotations

from typing import Any

import numpy as np

from fem import FEMModel, PlateGeometry
from models import (
    BlackHole,
    HarmonicPointForce,
    Material,
    MeshConfig,
    Plate,
    Sensor,
)
from solvers import (
    FRFSolver,
    ModalBasis,
    ModalFRFSolver,
    ModalSolver,
    RayleighDamping,
)


class SimulationManager:
    """
    Chef d'orchestre du projet.

    Rôle :
    - stocker les paramètres du cas
    - construire les objets physiques
    - construire le modèle EF
    - lancer les analyses modales et FRF
    - fournir des données prêtes pour l'interface
    """

    def __init__(self, load_defaults: bool = True) -> None:
        self.material_params: dict[str, Any] = {}
        self.plate_params: dict[str, Any] = {}
        self.black_hole_params: dict[str, Any] = {}
        self.excitation_params: dict[str, Any] = {}
        self.mesh_params: dict[str, Any] = {}
        self.sensors: dict[str, dict[str, Any]] = {}

        self._model_cache: dict[tuple[bool, bool], FEMModel] = {}
        self._modal_basis_cache: dict[tuple[bool, bool, int], ModalBasis] = {}
        self.last_results: dict[str, Any] = {}

        if load_defaults:
            self.load_default_case()

    # ------------------------------------------------------------------
    # Gestion des paramètres
    # ------------------------------------------------------------------

    def invalidate_cache(self) -> None:
        self._model_cache.clear()
        self._modal_basis_cache.clear()

    def reset(self) -> None:
        self.material_params.clear()
        self.plate_params.clear()
        self.black_hole_params.clear()
        self.excitation_params.clear()
        self.mesh_params.clear()
        self.sensors.clear()
        self.last_results.clear()
        self.invalidate_cache()

    def load_default_case(self) -> None:
        """
        Cas de départ cohérent avec ton étude actuelle.
        """
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
            boundary_condition="simply_supported",
            name="Rectangular plate",
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

        self.set_excitation(
            x=0.10,
            y=0.10,
            amplitude=1.0,
            frequency_start=50.0,
            frequency_end=700.0,
            n_points=400,
            phase_deg=0.0,
            direction="w",
        )

        self.set_mesh(
            element_size=0.02,
            refine_near_black_hole=True,
            refinement_radius=0.08,
            refinement_element_size=0.005,
            element_type="tri3",
        )

        self.set_sensor(
            name="S1",
            x=0.35,
            y=0.25,
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
        boundary_condition: str = "simply_supported",
        name: str = "Rectangular plate",
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

    def disable_black_hole(self) -> None:
        if not self.black_hole_params:
            return
        self.black_hole_params["enabled"] = False
        self.invalidate_cache()

    def enable_black_hole(self) -> None:
        if not self.black_hole_params:
            return
        self.black_hole_params["enabled"] = True
        self.invalidate_cache()

    def set_excitation(
        self,
        x: float,
        y: float,
        amplitude: float,
        frequency_start: float,
        frequency_end: float,
        n_points: int = 400,
        phase_deg: float = 0.0,
        direction: str = "w",
    ) -> None:
        self.excitation_params = {
            "x": x,
            "y": y,
            "amplitude": amplitude,
            "frequency_start": frequency_start,
            "frequency_end": frequency_end,
            "n_points": n_points,
            "phase_deg": phase_deg,
            "direction": direction,
        }

    def set_mesh(
        self,
        element_size: float,
        refine_near_black_hole: bool = True,
        refinement_radius: float = 0.08,
        refinement_element_size: float = 0.005,
        element_type: str = "tri3",
    ) -> None:
        self.mesh_params = {
            "element_size": element_size,
            "refine_near_black_hole": refine_near_black_hole,
            "refinement_radius": refinement_radius,
            "refinement_element_size": refinement_element_size,
            "element_type": element_type,
        }
        self.invalidate_cache()

    def set_sensor(
        self,
        name: str,
        x: float,
        y: float,
        response_type: str = "displacement",
    ) -> None:
        self.sensors[name] = {
            "x": x,
            "y": y,
            "name": name,
            "response_type": response_type,
        }

    def add_sensor(
        self,
        name: str,
        x: float,
        y: float,
        response_type: str = "displacement",
    ) -> None:
        self.set_sensor(name=name, x=x, y=y, response_type=response_type)

    def remove_sensor(self, name: str) -> None:
        if name in self.sensors:
            del self.sensors[name]

    def clear_sensors(self) -> None:
        self.sensors.clear()

    def list_sensor_names(self) -> list[str]:
        return list(self.sensors.keys())

    # ------------------------------------------------------------------
    # Constructions internes
    # ------------------------------------------------------------------

    def build_material(self) -> Material:
        if not self.material_params:
            raise RuntimeError("Les paramètres matériau ne sont pas définis.")
        return Material(**self.material_params)

    def build_black_hole(self) -> BlackHole | None:
        if not self.black_hole_params:
            return None
        return BlackHole(**self.black_hole_params)

    def build_plate(self) -> Plate:
        if not self.plate_params:
            raise RuntimeError("Les paramètres plaque ne sont pas définis.")

        material = self.build_material()
        black_hole = self.build_black_hole()

        plate = Plate(
            length_x=self.plate_params["length_x"],
            length_y=self.plate_params["length_y"],
            thickness=self.plate_params["thickness"],
            material=material,
            boundary_condition=self.plate_params["boundary_condition"],
            black_hole=black_hole,
            name=self.plate_params["name"],
        )

        plate.validate_black_hole_inside()
        return plate

    def build_excitation(self) -> HarmonicPointForce:
        if not self.excitation_params:
            raise RuntimeError("Les paramètres d'excitation ne sont pas définis.")
        return HarmonicPointForce(**self.excitation_params)

    def build_sensor(self, name: str) -> Sensor:
        if name not in self.sensors:
            raise KeyError(f"Capteur inconnu : '{name}'")
        return Sensor(**self.sensors[name])

    def build_mesh_config(self) -> MeshConfig:
        if not self.mesh_params:
            raise RuntimeError("Les paramètres de maillage ne sont pas définis.")
        return MeshConfig(**self.mesh_params)

    def build_geometry(self) -> PlateGeometry:
        plate = self.build_plate()
        return PlateGeometry(plate)

    # ------------------------------------------------------------------
    # Accès géométrie / maillage
    # ------------------------------------------------------------------

    def get_thickness_field(
        self,
        nx: int = 301,
        ny: int = 241,
        use_black_hole: bool = True,
    ) -> dict[str, np.ndarray]:
        geometry = self.build_geometry()
        X, Y = geometry.make_grid(nx=nx, ny=ny)
        H = geometry.thickness_field(X, Y, use_black_hole=use_black_hole)

        return {
            "X": X,
            "Y": Y,
            "H": H,
        }

    def get_flexural_rigidity_field(
        self,
        nx: int = 301,
        ny: int = 241,
        use_black_hole: bool = True,
    ) -> dict[str, np.ndarray]:
        geometry = self.build_geometry()
        X, Y = geometry.make_grid(nx=nx, ny=ny)
        D = geometry.flexural_rigidity_field(X, Y, use_black_hole=use_black_hole)

        return {
            "X": X,
            "Y": Y,
            "D": D,
        }

    def get_fem_model(
        self,
        use_black_hole: bool = True,
        refine_with_black_hole_region: bool = True,
        rebuild: bool = False,
    ) -> FEMModel:
        key = (use_black_hole, refine_with_black_hole_region)

        if (not rebuild) and key in self._model_cache:
            return self._model_cache[key]

        plate = self.build_plate()
        mesh_config = self.build_mesh_config()

        model = FEMModel(
            plate=plate,
            mesh_config=mesh_config,
            use_black_hole=use_black_hole,
            refine_with_black_hole_region=refine_with_black_hole_region,
        )
        model.build()

        self._model_cache[key] = model
        return model

    def get_mesh_preview(
        self,
        use_black_hole: bool = True,
        refine_with_black_hole_region: bool = True,
        rebuild: bool = False,
    ):
        model = self.get_fem_model(
            use_black_hole=use_black_hole,
            refine_with_black_hole_region=refine_with_black_hole_region,
            rebuild=rebuild,
        )
        return model.mesh

    def get_modal_basis(
        self,
        n_modes: int = 30,
        use_black_hole: bool = True,
        refine_with_black_hole_region: bool = True,
        rebuild: bool = False,
    ) -> ModalBasis:
        key = (use_black_hole, refine_with_black_hole_region, n_modes)

        if (not rebuild) and key in self._modal_basis_cache:
            return self._modal_basis_cache[key]

        model = self.get_fem_model(
            use_black_hole=use_black_hole,
            refine_with_black_hole_region=refine_with_black_hole_region,
            rebuild=rebuild,
        )

        solver = ModalSolver(model)
        basis = solver.solve_basis(n_modes=n_modes)

        self._modal_basis_cache[key] = basis
        return basis

    # ------------------------------------------------------------------
    # Analyses
    # ------------------------------------------------------------------

    def run_modal_analysis(
        self,
        n_modes: int = 6,
        use_black_hole: bool = True,
        refine_with_black_hole_region: bool = True,
        rebuild: bool = False,
    ) -> dict[str, Any]:
        model = self.get_fem_model(
            use_black_hole=use_black_hole,
            refine_with_black_hole_region=refine_with_black_hole_region,
            rebuild=rebuild,
        )

        basis = self.get_modal_basis(
            n_modes=n_modes,
            use_black_hole=use_black_hole,
            refine_with_black_hole_region=refine_with_black_hole_region,
            rebuild=rebuild,
        )

        result = {
            "model": model,
            "frequencies_hz": basis.frequencies_hz,
            "modes_full": basis.modes_full,
            "modes_free": basis.modes_free,
            "omegas_rad_s": basis.omegas_rad_s,
            "modal_masses": basis.modal_masses,
        }

        self.last_results["modal"] = result
        return result

    def compare_modal(
        self,
        n_modes: int = 6,
        refine_with_black_hole_region: bool = True,
        rebuild: bool = False,
    ) -> dict[str, Any]:
        ref = self.run_modal_analysis(
            n_modes=n_modes,
            use_black_hole=False,
            refine_with_black_hole_region=refine_with_black_hole_region,
            rebuild=rebuild,
        )

        bh = self.run_modal_analysis(
            n_modes=n_modes,
            use_black_hole=True,
            refine_with_black_hole_region=refine_with_black_hole_region,
            rebuild=rebuild,
        )

        n = min(len(ref["frequencies_hz"]), len(bh["frequencies_hz"]))
        freqs_ref = ref["frequencies_hz"][:n]
        freqs_bh = bh["frequencies_hz"][:n]
        delta_pct = 100.0 * (freqs_bh - freqs_ref) / freqs_ref

        result = {
            "reference": ref,
            "black_hole": bh,
            "freqs_ref_hz": freqs_ref,
            "freqs_bh_hz": freqs_bh,
            "delta_pct": delta_pct,
        }

        self.last_results["modal_comparison"] = result
        return result

    def run_frf_direct(
        self,
        sensor_name: str,
        use_black_hole: bool = True,
        refine_with_black_hole_region: bool = True,
        damping_ratio: float = 0.01,
        damping_freq1_hz: float = 140.0,
        damping_freq2_hz: float = 500.0,
        rebuild: bool = False,
    ) -> dict[str, Any]:
        model = self.get_fem_model(
            use_black_hole=use_black_hole,
            refine_with_black_hole_region=refine_with_black_hole_region,
            rebuild=rebuild,
        )

        excitation = self.build_excitation()
        sensor = self.build_sensor(sensor_name)

        damping = RayleighDamping.from_modal_damping_ratio(
            zeta=damping_ratio,
            freq1_hz=damping_freq1_hz,
            freq2_hz=damping_freq2_hz,
        )

        solver = FRFSolver(model)
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

        self.last_results["frf_direct"] = result
        return result

    def run_frf_modal(
        self,
        sensor_name: str,
        use_black_hole: bool = True,
        refine_with_black_hole_region: bool = True,
        n_modes: int = 30,
        damping_ratio: float = 0.01,
        rebuild: bool = False,
    ) -> dict[str, Any]:
        model = self.get_fem_model(
            use_black_hole=use_black_hole,
            refine_with_black_hole_region=refine_with_black_hole_region,
            rebuild=rebuild,
        )

        excitation = self.build_excitation()
        sensor = self.build_sensor(sensor_name)

        basis = self.get_modal_basis(
            n_modes=n_modes,
            use_black_hole=use_black_hole,
            refine_with_black_hole_region=refine_with_black_hole_region,
            rebuild=rebuild,
        )

        solver = ModalFRFSolver(model)
        frf_result = solver.solve(
            excitation=excitation,
            sensor=sensor,
            n_modes=n_modes,
            damping_ratio=damping_ratio,
            modal_basis=basis,
        )

        result = {
            "model": model,
            "sensor": sensor,
            "excitation": excitation,
            "modal_basis": basis,
            "frf_result": frf_result,
        }

        self.last_results["frf_modal"] = result
        return result

    def compare_frf_modal(
        self,
        sensor_name: str,
        n_modes: int = 30,
        damping_ratio: float = 0.01,
        refine_with_black_hole_region: bool = True,
        rebuild: bool = False,
    ) -> dict[str, Any]:
        ref = self.run_frf_modal(
            sensor_name=sensor_name,
            use_black_hole=False,
            refine_with_black_hole_region=refine_with_black_hole_region,
            n_modes=n_modes,
            damping_ratio=damping_ratio,
            rebuild=rebuild,
        )

        bh = self.run_frf_modal(
            sensor_name=sensor_name,
            use_black_hole=True,
            refine_with_black_hole_region=refine_with_black_hole_region,
            n_modes=n_modes,
            damping_ratio=damping_ratio,
            rebuild=rebuild,
        )

        result = {
            "reference": ref,
            "black_hole": bh,
        }

        self.last_results["frf_modal_comparison"] = result
        return result

    # ------------------------------------------------------------------
    # Résumés utiles pour l'interface
    # ------------------------------------------------------------------

    def get_case_summary(self) -> dict[str, Any]:
        return {
            "material": dict(self.material_params),
            "plate": dict(self.plate_params),
            "black_hole": dict(self.black_hole_params),
            "excitation": dict(self.excitation_params),
            "mesh": dict(self.mesh_params),
            "sensors": {name: dict(params) for name, params in self.sensors.items()},
        }


if __name__ == "__main__":
    manager = SimulationManager(load_defaults=True)

    print("=== Résumé du cas ===")
    print(manager.get_case_summary())

    modal = manager.run_modal_analysis(n_modes=6, use_black_hole=True)
    print("\n=== Fréquences propres avec VABH ===")
    for i, f in enumerate(modal["frequencies_hz"], start=1):
        print(f"Mode {i:02d} : {f:.3f} Hz")

    frf_modal = manager.run_frf_modal(
        sensor_name="S1",
        use_black_hole=True,
        n_modes=30,
        damping_ratio=0.01,
    )

    frf_result = frf_modal["frf_result"]
    print("\n=== FRF modale ===")
    print("Nombre de points fréquentiels :", len(frf_result.frequencies_hz))
    print("Nombre de modes utilisés      :", frf_result.n_modes_used)
    print("Noeud excitation              :", frf_result.excitation_node_id)
    print("Noeud capteur                 :", frf_result.sensor_node_id)