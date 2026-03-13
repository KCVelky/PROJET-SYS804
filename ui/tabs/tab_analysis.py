# ui/tabs/tab_analysis.py

from __future__ import annotations

import matplotlib.tri as mtri
import numpy as np

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QTextEdit,
    QApplication,
)
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from core import SimulationManager
from ui.widgets import MplCanvas


class AnalysisTab(QWidget):
    def __init__(self, manager: SimulationManager, apply_callback) -> None:
        super().__init__()
        self.manager = manager
        self.apply_callback = apply_callback

        self.last_modal_result = None
        self.last_modal_comparison = None
        self.last_frf_result = None
        self.last_frf_comparison = None

        root = QHBoxLayout(self)

        left_col = QVBoxLayout()
        right_col = QVBoxLayout()

        controls_group = QGroupBox("Pilotage")
        controls_form = QFormLayout(controls_group)

        self.use_black_hole_checkbox = QCheckBox("Inclure le trou noir dans l'analyse")
        self.use_black_hole_checkbox.setChecked(True)

        self.n_modes_spin = QSpinBox()
        self.n_modes_spin.setRange(1, 500)
        self.n_modes_spin.setValue(30)

        self.mode_to_plot_spin = QSpinBox()
        self.mode_to_plot_spin.setRange(1, 500)
        self.mode_to_plot_spin.setValue(1)

        self.damping_ratio = QDoubleSpinBox()
        self.damping_ratio.setRange(0.0, 1.0)
        self.damping_ratio.setDecimals(4)
        self.damping_ratio.setValue(0.01)

        self.grid_nx = QSpinBox()
        self.grid_nx.setRange(20, 2000)
        self.grid_nx.setValue(301)

        self.grid_ny = QSpinBox()
        self.grid_ny.setRange(20, 2000)
        self.grid_ny.setValue(241)

        self.sensor_combo = QComboBox()

        controls_form.addRow(self.use_black_hole_checkbox)
        controls_form.addRow("Nombre de modes", self.n_modes_spin)
        controls_form.addRow("Mode à afficher", self.mode_to_plot_spin)
        controls_form.addRow("Amortissement modal ζ", self.damping_ratio)
        controls_form.addRow("Grille nx", self.grid_nx)
        controls_form.addRow("Grille ny", self.grid_ny)
        controls_form.addRow("Capteur", self.sensor_combo)

        buttons_group = QGroupBox("Actions")
        buttons_layout = QVBoxLayout(buttons_group)

        self.btn_thickness = QPushButton("Tracer champ d'épaisseur")
        self.btn_rigidity = QPushButton("Tracer rigidité de flexion")
        self.btn_mesh = QPushButton("Tracer maillage")
        self.btn_modal = QPushButton("Tracer mode propre")
        self.btn_modal_compare = QPushButton("Comparer les fréquences propres")
        self.btn_frf = QPushButton("Tracer FRF modale")
        self.btn_frf_compare = QPushButton("Comparer les FRF modales")

        for btn in [
            self.btn_thickness,
            self.btn_rigidity,
            self.btn_mesh,
            self.btn_modal,
            self.btn_modal_compare,
            self.btn_frf,
            self.btn_frf_compare,
        ]:
            buttons_layout.addWidget(btn)

        buttons_layout.addStretch(1)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        left_col.addWidget(controls_group)
        left_col.addWidget(buttons_group)
        left_col.addWidget(self.log_box)

        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        right_col.addWidget(self.toolbar)
        right_col.addWidget(self.canvas)

        root.addLayout(left_col, 0)
        root.addLayout(right_col, 1)

        self.refresh_sensor_list()
        self._connect_signals()

    def _connect_signals(self) -> None:
        self.btn_thickness.clicked.connect(self.plot_thickness)
        self.btn_rigidity.clicked.connect(self.plot_rigidity)
        self.btn_mesh.clicked.connect(self.plot_mesh)
        self.btn_modal.clicked.connect(self.plot_mode)
        self.btn_modal_compare.clicked.connect(self.compare_modal)
        self.btn_frf.clicked.connect(self.plot_frf_modal)
        self.btn_frf_compare.clicked.connect(self.compare_frf_modal)

    def refresh_sensor_list(self) -> None:
        current = self.sensor_combo.currentText()
        self.sensor_combo.clear()
        names = self.manager.list_sensor_names()
        self.sensor_combo.addItems(names)
        if current and current in names:
            self.sensor_combo.setCurrentText(current)

    def log(self, text: str) -> None:
        self.log_box.append(text)

    def _prepare(self) -> bool:
        ok = self.apply_callback()
        if not ok:
            return False
        self.refresh_sensor_list()
        QApplication.processEvents()
        return True

    def _overlay_black_hole_boundaries(self, ax) -> None:
        try:
            geometry = self.manager.build_geometry()
            outer = geometry.black_hole_boundary()
            inner = geometry.truncation_boundary()
            if outer is not None:
                ax.plot(outer[0] * 1e3, outer[1] * 1e3, "r--", linewidth=2, label="Rayon extérieur")
            if inner is not None:
                ax.plot(inner[0] * 1e3, inner[1] * 1e3, "k--", linewidth=2, label="Troncature")
        except Exception:
            pass

    def plot_thickness(self) -> None:
        if not self._prepare():
            return

        self.log("Calcul du champ d'épaisseur...")
        field = self.manager.get_thickness_field(
            nx=self.grid_nx.value(),
            ny=self.grid_ny.value(),
            use_black_hole=self.use_black_hole_checkbox.isChecked(),
        )

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        c = ax.contourf(field["X"] * 1e3, field["Y"] * 1e3, field["H"] * 1e3, levels=60)
        self.canvas.figure.colorbar(c, ax=ax, label="Épaisseur [mm]")

        self._overlay_black_hole_boundaries(ax)

        ax.set_title("Champ d'épaisseur")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        self.canvas.draw_idle()

    def plot_rigidity(self) -> None:
        if not self._prepare():
            return

        self.log("Calcul du champ de rigidité...")
        field = self.manager.get_flexural_rigidity_field(
            nx=self.grid_nx.value(),
            ny=self.grid_ny.value(),
            use_black_hole=self.use_black_hole_checkbox.isChecked(),
        )

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        c = ax.contourf(field["X"] * 1e3, field["Y"] * 1e3, field["D"], levels=60)
        self.canvas.figure.colorbar(c, ax=ax, label="Rigidité D [N.m]")

        self._overlay_black_hole_boundaries(ax)

        ax.set_title("Rigidité de flexion locale")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        self.canvas.draw_idle()

    def plot_mesh(self) -> None:
        if not self._prepare():
            return

        self.log("Génération du maillage...")
        mesh = self.manager.get_mesh_preview(
            use_black_hole=self.use_black_hole_checkbox.isChecked(),
            refine_with_black_hole_region=True,
            rebuild=True,
        )

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        triang = mtri.Triangulation(
            mesh.nodes[:, 0] * 1e3,
            mesh.nodes[:, 1] * 1e3,
            triangles=mesh.elements,
        )
        ax.triplot(triang, linewidth=0.5)

        self._overlay_black_hole_boundaries(ax)

        ax.set_title(f"Maillage | nœuds = {mesh.n_nodes} | éléments = {mesh.n_elements}")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")
        self.canvas.draw_idle()

        self.log(f"Maillage généré : {mesh.n_nodes} nœuds, {mesh.n_elements} éléments.")

    def plot_mode(self) -> None:
        if not self._prepare():
            return

        n_modes = max(self.n_modes_spin.value(), self.mode_to_plot_spin.value())

        self.log("Calcul modal en cours...")
        result = self.manager.run_modal_analysis(
            n_modes=n_modes,
            use_black_hole=self.use_black_hole_checkbox.isChecked(),
            refine_with_black_hole_region=True,
            rebuild=True,
        )
        self.last_modal_result = result

        mode_id = self.mode_to_plot_spin.value() - 1
        freqs = result["frequencies_hz"]
        if mode_id >= len(freqs):
            self.log("Le mode demandé n'est pas disponible.")
            return

        model = result["model"]
        mode_vec = result["modes_full"][:, mode_id]
        w = mode_vec[0::3]
        w = w / np.max(np.abs(w))

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)

        triang = mtri.Triangulation(
            model.mesh.nodes[:, 0] * 1e3,
            model.mesh.nodes[:, 1] * 1e3,
            triangles=model.mesh.elements,
        )
        c = ax.tricontourf(triang, w, levels=40)
        self.canvas.figure.colorbar(c, ax=ax, label="Amplitude modale normalisée")
        ax.triplot(triang, color="k", linewidth=0.25, alpha=0.3)

        ax.set_title(f"Mode {mode_id + 1} | f = {freqs[mode_id]:.3f} Hz")
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

        self.log("Fréquences propres :")
        for i, f in enumerate(freqs, start=1):
            self.log(f"  Mode {i:02d} : {f:.3f} Hz")

    def compare_modal(self) -> None:
        if not self._prepare():
            return

        self.log("Comparaison modale uniforme / VABH...")
        result = self.manager.compare_modal(
            n_modes=self.n_modes_spin.value(),
            refine_with_black_hole_region=True,
            rebuild=True,
        )
        self.last_modal_comparison = result

        ref = result["freqs_ref_hz"]
        bh = result["freqs_bh_hz"]
        delta = result["delta_pct"]
        modes = np.arange(1, len(ref) + 1)

        self.canvas.figure.clear()
        ax = self.canvas.figure.add_subplot(111)
        ax.plot(modes, ref, "o-", label="Plaque uniforme")
        ax.plot(modes, bh, "s-", label="Plaque avec VABH")
        ax.set_xlabel("Numéro de mode")
        ax.set_ylabel("Fréquence [Hz]")
        ax.set_title("Comparaison des fréquences propres")
        ax.grid(True, alpha=0.3)
        ax.legend()
        self.canvas.draw_idle()

        self.log("Mode | Réf [Hz] | VABH [Hz] | Écart [%]")
        self.log("-" * 42)
        for i in range(len(ref)):
            self.log(f"{i+1:>4d} | {ref[i]:>8.3f} | {bh[i]:>9.3f} | {delta[i]:>8.3f}")

    def plot_frf_modal(self) -> None:
        if not self._prepare():
            return

        sensor_name = self.sensor_combo.currentText()
        if not sensor_name:
            self.log("Aucun capteur disponible.")
            return

        self.log("Calcul de la FRF modale...")
        result = self.manager.run_frf_modal(
            sensor_name=sensor_name,
            use_black_hole=self.use_black_hole_checkbox.isChecked(),
            refine_with_black_hole_region=True,
            n_modes=self.n_modes_spin.value(),
            damping_ratio=self.damping_ratio.value(),
            rebuild=True,
        )
        self.last_frf_result = result
        frf = result["frf_result"]

        self.canvas.figure.clear()
        ax1 = self.canvas.figure.add_subplot(211)
        ax2 = self.canvas.figure.add_subplot(212)

        ax1.semilogy(frf.frequencies_hz, frf.magnitude)
        ax1.set_ylabel("|FRF|")
        ax1.set_title(
            f"FRF modale | capteur = {sensor_name} | modes = {frf.n_modes_used}"
        )
        ax1.grid(True, alpha=0.3)

        ax2.plot(frf.frequencies_hz, frf.phase_deg)
        ax2.set_xlabel("Fréquence [Hz]")
        ax2.set_ylabel("Phase [deg]")
        ax2.grid(True, alpha=0.3)

        self.canvas.draw_idle()

        self.log(
            f"FRF modale calculée | n_points = {len(frf.frequencies_hz)} | "
            f"n_modes = {frf.n_modes_used}"
        )
        self.log(
            f"Noeud excitation = {frf.excitation_node_id} | "
            f"écart = {frf.excitation_distance_m * 1e3:.3f} mm"
        )
        self.log(
            f"Noeud capteur = {frf.sensor_node_id} | "
            f"écart = {frf.sensor_distance_m * 1e3:.3f} mm"
        )

    def compare_frf_modal(self) -> None:
        if not self._prepare():
            return

        sensor_name = self.sensor_combo.currentText()
        if not sensor_name:
            self.log("Aucun capteur disponible.")
            return

        self.log("Comparaison des FRF modales uniforme / VABH...")
        result = self.manager.compare_frf_modal(
            sensor_name=sensor_name,
            n_modes=self.n_modes_spin.value(),
            damping_ratio=self.damping_ratio.value(),
            refine_with_black_hole_region=True,
            rebuild=True,
        )
        self.last_frf_comparison = result

        ref = result["reference"]["frf_result"]
        bh = result["black_hole"]["frf_result"]

        self.canvas.figure.clear()
        ax1 = self.canvas.figure.add_subplot(211)
        ax2 = self.canvas.figure.add_subplot(212)

        ax1.semilogy(ref.frequencies_hz, ref.magnitude, label="Plaque uniforme")
        ax1.semilogy(bh.frequencies_hz, bh.magnitude, label="Plaque avec VABH")
        ax1.set_ylabel("|FRF|")
        ax1.set_title("Comparaison des FRF modales")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(ref.frequencies_hz, ref.phase_deg, label="Plaque uniforme")
        ax2.plot(bh.frequencies_hz, bh.phase_deg, label="Plaque avec VABH")
        ax2.set_xlabel("Fréquence [Hz]")
        ax2.set_ylabel("Phase [deg]")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        self.canvas.draw_idle()

        self.log("Comparaison FRF modale terminée.")