from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from models.plate import Plate
from solid3d.fem_model_3d import Solid3DFEMModel
from solid3d.gmsh_mesher_3d import Solid3DMeshOptions
from solid3d.modal_solver_3d import ModalBasis3D, ModalSolver3D
from solid3d.pyvista_viewer_3d import Solid3DMeshViewer


@dataclass
class ModeComparison3D:
    mode_number: int
    frequency_uniform_hz: float
    frequency_abh_hz: float
    relative_shift_pct: float
    localization_uniform: float
    localization_abh: float
    localization_gain: float


@dataclass
class ModalComparisonResult3D:
    plate_uniform: Plate
    plate_abh: Plate
    mesh_options_uniform: Solid3DMeshOptions
    mesh_options_abh: Solid3DMeshOptions
    model_uniform: Solid3DFEMModel
    model_abh: Solid3DFEMModel
    basis_uniform: ModalBasis3D
    basis_abh: ModalBasis3D
    mode_comparisons: list[ModeComparison3D]

    @property
    def frequencies_uniform_hz(self) -> np.ndarray:
        return self.basis_uniform.frequencies_hz

    @property
    def frequencies_abh_hz(self) -> np.ndarray:
        return self.basis_abh.frequencies_hz


class ModalComparison3D:
    def __init__(
        self,
        plate: Plate,
        mesh_options: Solid3DMeshOptions,
        verbose: bool = True,
    ) -> None:
        if plate.black_hole is None:
            raise ValueError(
                "La plaque doit contenir une définition de trou noir pour faire la comparaison 3D."
            )
        self.plate = deepcopy(plate)
        self.mesh_options = mesh_options
        self.verbose = verbose

    def _make_plate_copy(self, use_black_hole: bool) -> Plate:
        plate_case = deepcopy(self.plate)
        if plate_case.black_hole is not None:
            plate_case.black_hole.enabled = bool(use_black_hole)
        return plate_case

    def _make_mesh_options_copy(self, use_black_hole: bool) -> Solid3DMeshOptions:
        if self.mesh_options.save_msh_path is None:
            return replace(self.mesh_options)
        src = Path(self.mesh_options.save_msh_path)
        suffix = "_abh" if use_black_hole else "_uniform"
        dst = src.with_name(f"{src.stem}{suffix}{src.suffix}")
        return replace(self.mesh_options, save_msh_path=str(dst))

    def _solve_case(
        self,
        use_black_hole: bool,
        n_modes: int,
    ) -> tuple[Plate, Solid3DMeshOptions, Solid3DFEMModel, ModalBasis3D]:
        plate_case = self._make_plate_copy(use_black_hole=use_black_hole)
        mesh_options_case = self._make_mesh_options_copy(use_black_hole=use_black_hole)

        model = Solid3DFEMModel(
            plate=plate_case,
            mesh_options=mesh_options_case,
            use_black_hole=use_black_hole,
            verbose=self.verbose,
        )
        model.build()

        solver = ModalSolver3D(model, verbose=self.verbose)
        basis = solver.solve_basis(n_modes=n_modes)
        return plate_case, mesh_options_case, model, basis

    @staticmethod
    def _localization_index(model: Solid3DFEMModel, mode_vector_full: np.ndarray) -> float:
        bh = model.plate.black_hole
        if bh is None or model.mesh is None:
            return float("nan")

        coords = model.mesh.points
        r = np.sqrt((coords[:, 0] - bh.xc) ** 2 + (coords[:, 1] - bh.yc) ** 2)
        local_mask = r <= bh.radius
        if not np.any(local_mask):
            return 0.0

        disp = mode_vector_full.reshape(-1, 3)
        amp2 = np.sum(disp ** 2, axis=1)
        total = float(np.sum(amp2))
        if total <= 1e-30:
            return 0.0

        local = float(np.sum(amp2[local_mask]))
        return local / total

    def run(self, n_modes: int = 6) -> ModalComparisonResult3D:
        if self.verbose:
            print("=== Cas 1 : plaque uniforme 3D ===")
        plate_uniform, mesh_uniform, model_uniform, basis_uniform = self._solve_case(
            use_black_hole=False,
            n_modes=n_modes,
        )

        if self.verbose:
            print("=== Cas 2 : plaque avec ABH 3D ===")
        plate_abh, mesh_abh, model_abh, basis_abh = self._solve_case(
            use_black_hole=True,
            n_modes=n_modes,
        )

        n = min(basis_uniform.frequencies_hz.size, basis_abh.frequencies_hz.size)
        comparisons: list[ModeComparison3D] = []

        for i in range(n):
            f_ref = float(basis_uniform.frequencies_hz[i])
            f_abh = float(basis_abh.frequencies_hz[i])
            shift_pct = 100.0 * (f_abh - f_ref) / f_ref if abs(f_ref) > 1e-30 else np.nan

            loc_ref = self._localization_index(model_uniform, basis_uniform.modes_full[:, i])
            loc_abh = self._localization_index(model_abh, basis_abh.modes_full[:, i])
            gain = loc_abh / loc_ref if loc_ref > 1e-30 else np.nan

            comparisons.append(
                ModeComparison3D(
                    mode_number=i + 1,
                    frequency_uniform_hz=f_ref,
                    frequency_abh_hz=f_abh,
                    relative_shift_pct=shift_pct,
                    localization_uniform=loc_ref,
                    localization_abh=loc_abh,
                    localization_gain=gain,
                )
            )

        return ModalComparisonResult3D(
            plate_uniform=plate_uniform,
            plate_abh=plate_abh,
            mesh_options_uniform=mesh_uniform,
            mesh_options_abh=mesh_abh,
            model_uniform=model_uniform,
            model_abh=model_abh,
            basis_uniform=basis_uniform,
            basis_abh=basis_abh,
            mode_comparisons=comparisons,
        )

    @staticmethod
    def print_summary(result: ModalComparisonResult3D) -> None:
        print("\n=== Comparaison modale 3D : plaque uniforme vs plaque avec ABH ===")
        print(
            f"{'Mode':>4} | {'Uniforme [Hz]':>14} | {'ABH [Hz]':>12} | {'Δf [%]':>10} | "
            f"{'Loc. unif.':>10} | {'Loc. ABH':>9} | {'Gain':>9}"
        )
        print("-" * 92)
        for item in result.mode_comparisons:
            print(
                f"{item.mode_number:>4d} | "
                f"{item.frequency_uniform_hz:>14.3f} | "
                f"{item.frequency_abh_hz:>12.3f} | "
                f"{item.relative_shift_pct:>10.3f} | "
                f"{item.localization_uniform:>10.4f} | "
                f"{item.localization_abh:>9.4f} | "
                f"{item.localization_gain:>9.3f}"
            )

    @staticmethod
    def plot_frequency_comparison(result: ModalComparisonResult3D) -> None:
        freqs_ref = result.frequencies_uniform_hz
        freqs_abh = result.frequencies_abh_hz
        n = min(len(freqs_ref), len(freqs_abh))
        modes = np.arange(1, n + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(modes, freqs_ref[:n], "o-", label="Plaque uniforme 3D")
        ax.plot(modes, freqs_abh[:n], "s-", label="Plaque avec ABH 3D")
        ax.set_xlabel("Numéro de mode")
        ax.set_ylabel("Fréquence propre [Hz]")
        ax.set_title("Comparaison modale 3D")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

    @staticmethod
    def plot_relative_shift(result: ModalComparisonResult3D) -> None:
        shift_pct = np.array([item.relative_shift_pct for item in result.mode_comparisons], dtype=float)
        modes = np.arange(1, len(shift_pct) + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(modes, shift_pct)
        ax.set_xlabel("Numéro de mode")
        ax.set_ylabel("Variation relative [%]")
        ax.set_title("Impact relatif de l'ABH sur les fréquences propres 3D")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    @staticmethod
    def plot_localization_comparison(result: ModalComparisonResult3D) -> None:
        modes = np.arange(1, len(result.mode_comparisons) + 1)
        loc_ref = np.array([item.localization_uniform for item in result.mode_comparisons], dtype=float)
        loc_abh = np.array([item.localization_abh for item in result.mode_comparisons], dtype=float)

        fig, ax = plt.subplots(figsize=(8, 5))
        width = 0.35
        ax.bar(modes - width / 2.0, loc_ref, width=width, label="Uniforme")
        ax.bar(modes + width / 2.0, loc_abh, width=width, label="ABH")
        ax.set_xlabel("Numéro de mode")
        ax.set_ylabel("Indice de localisation dans la zone ABH [-]")
        ax.set_title("Concentration modale dans la zone du trou noir")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()

    @staticmethod
    def show_mode_pair(
        result: ModalComparisonResult3D,
        mode_number: int,
        color_by: str = "umag",
        scale: float | None = None,
    ) -> None:
        idx = mode_number - 1
        if idx < 0:
            raise ValueError("mode_number doit être >= 1.")
        if idx >= result.basis_uniform.modes_full.shape[1] or idx >= result.basis_abh.modes_full.shape[1]:
            raise ValueError("mode_number dépasse le nombre de modes disponibles.")

        Solid3DMeshViewer.show_mode_shape(
            mesh=result.model_uniform.mesh,
            mode_vector=result.basis_uniform.modes_full[:, idx],
            title=f"Mode {mode_number} - plaque uniforme 3D",
            color_by=color_by,
            scale=scale,
        )
        Solid3DMeshViewer.show_mode_shape(
            mesh=result.model_abh.mesh,
            mode_vector=result.basis_abh.modes_full[:, idx],
            title=f"Mode {mode_number} - plaque avec ABH 3D",
            color_by=color_by,
            scale=scale,
        )
