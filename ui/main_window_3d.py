from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.simulation_manager_3d import SimulationManager3D

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
    HAS_PYVISTA = True
except Exception:
    pv = None
    QtInteractor = None
    HAS_PYVISTA = False

try:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    HAS_MPL = True
except Exception:
    FigureCanvas = None
    Figure = None
    HAS_MPL = False


class PlotPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        if HAS_MPL:
            self.figure = Figure(figsize=(5, 3), tight_layout=True)
            self.canvas = FigureCanvas(self.figure)
            self.ax = self.figure.add_subplot(111)
            layout.addWidget(self.canvas)
        else:
            self.figure = None
            self.canvas = None
            self.ax = None
            label = QLabel("Matplotlib non disponible.")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(
                "QLabel { color: #475569; background: white; border: 1px solid #d9dee7; "
                "border-radius: 8px; padding: 10px; }"
            )
            layout.addWidget(label)

    def plot_curve(self, x, y, title="", xlabel="", ylabel=""):
        if self.ax is None:
            return
        self.ax.clear()
        self.ax.plot(x, y)
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True, alpha=0.25)
        self.canvas.draw_idle()

    def plot_modes(self, freqs):
        if self.ax is None:
            return
        self.ax.clear()
        if len(freqs):
            self.ax.plot(range(1, len(freqs) + 1), freqs, marker="o")
        self.ax.set_title("Fréquences propres")
        self.ax.set_xlabel("Mode")
        self.ax.set_ylabel("Fréquence (Hz)")
        self.ax.grid(True, alpha=0.25)
        self.canvas.draw_idle()


class Viewer3D(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.plotter = None
        self._fallback_label = None
        if HAS_PYVISTA:
            self.plotter = QtInteractor(self)
            self.plotter.set_background("#f7f8fa")
            layout.addWidget(self.plotter)
        else:
            self._fallback_label = QLabel("PyVista / pyvistaqt non disponible.\nAperçu 3D désactivé.")
            self._fallback_label.setAlignment(Qt.AlignCenter)
            self._fallback_label.setStyleSheet(
                "QLabel { color: #475569; background: #f8fafc; font-size: 14px; "
                "border: 1px solid #d9dee7; border-radius: 10px; padding: 12px; }"
            )
            layout.addWidget(self._fallback_label)

    @staticmethod
    def _as_dataset(obj):
        if obj is None:
            return None
        if HAS_PYVISTA and isinstance(obj, pv.DataSet):
            return obj
        if hasattr(obj, "to_pyvista"):
            return obj.to_pyvista()
        return None

    def show_message(self, text: str):
        if self.plotter is not None:
            self.plotter.clear()
            self.plotter.add_text(text, position="upper_left", font_size=11, color="black")
        elif self._fallback_label is not None:
            self._fallback_label.setText(text)

    def show_mesh(self, mesh, show_edges=True):
        dataset = self._as_dataset(mesh)
        if dataset is None:
            self.show_message("Aucun dataset 3D à afficher.")
            return
        if self.plotter is None:
            return
        self.plotter.clear()
        self.plotter.add_mesh(dataset, color="lightgray", show_edges=show_edges, opacity=0.35)
        try:
            surface = dataset.extract_surface()
            self.plotter.add_mesh(surface, color="silver", show_edges=show_edges, opacity=0.7)
        except Exception:
            pass
        self.plotter.add_axes()
        self.plotter.reset_camera()

    def show_mode(self, mesh, scalars_name="mode_shape", show_edges=False):
        dataset = self._as_dataset(mesh)
        if dataset is None:
            self.show_message("Aucun mode 3D à afficher.")
            return
        if self.plotter is None:
            return
        self.plotter.clear()
        try:
            to_plot = dataset
            if "disp" in dataset.point_data:
                to_plot = dataset.warp_by_vector("disp", factor=1.0)
            surface = to_plot.extract_surface(algorithm="dataset_surface")
            self.plotter.add_mesh(surface, scalars=scalars_name, show_edges=show_edges)
            self.plotter.add_axes()
            self.plotter.reset_camera()
        except Exception as exc:
            self.show_message(f"Affichage mode impossible :\n{exc}")


class MainWindow3D(QMainWindow):
    def __init__(self, manager: SimulationManager3D | None = None):
        super().__init__()
        self.manager = manager or SimulationManager3D(load_defaults=True)
        self.mesh_result = None
        self.modal_result = None
        self.frf_result = None

        self.setWindowTitle("SYS804 - VBH 3D")
        self.resize(1500, 920)
        self.setStatusBar(QStatusBar())

        self._build_ui()
        self._apply_style()

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------
    def _build_ui(self):
        self._build_toolbar()

        root = QWidget()
        root_layout = QHBoxLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([430, 1070])

        root_layout.addWidget(splitter)
        self.setCentralWidget(root)

    def _build_toolbar(self):
        toolbar = QToolBar("Actions rapides")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        for text, slot in [
            ("Aperçu 3D", self.on_preview_mesh),
            ("Mailler", self.on_generate_mesh),
            ("Calcul modal", self.on_compute_modal),
            ("Calcul FRF", self.on_compute_frf),
        ]:
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            toolbar.addWidget(btn)

        toolbar.addSeparator()

        for text, slot in [
            ("Charger config", self.on_load_config),
            ("Sauver config", self.on_save_config),
        ]:
            btn = QPushButton(text)
            btn.setProperty("secondary", True)
            btn.clicked.connect(slot)
            toolbar.addWidget(btn)

    def _build_left_panel(self):
        tabs = QTabWidget()
        tabs.addTab(self._tab_geometry(), "Géométrie")
        tabs.addTab(self._tab_mesh(), "Maillage")
        tabs.addTab(self._tab_modal(), "Modal")
        tabs.addTab(self._tab_frf(), "FRF")
        tabs.addTab(self._tab_post(), "Post-traitement")
        return tabs

    def _build_right_panel(self):
        splitter = QSplitter(Qt.Vertical)
        self.viewer = Viewer3D()

        bottom_tabs = QTabWidget()
        bottom_tabs.addTab(self._results_tab(), "Résultats")
        bottom_tabs.addTab(self._plots_tab(), "Courbes")
        bottom_tabs.addTab(self._console_tab(), "Console")

        splitter.addWidget(self.viewer)
        splitter.addWidget(bottom_tabs)
        splitter.setSizes([620, 280])
        return splitter

    def _apply_style(self):
        self.setStyleSheet("""
            QMainWindow { background: #f2f4f7; }
            QWidget { color: #1f2937; background: #f2f4f7; }
            QLabel, QCheckBox, QRadioButton { color: #1f2937; background: transparent; }
            QGroupBox {
                color: #1f2937;
                font-weight: 600;
                border: 1px solid #d9dee7;
                border-radius: 10px;
                margin-top: 10px;
                padding-top: 8px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 4px;
                color: #334155;
                background: white;
            }
            QTabWidget::pane { border: 1px solid #d9dee7; background: white; border-radius: 10px; }
            QTabBar::tab {
                background: #e9edf3;
                color: #475569;
                padding: 8px 14px;
                margin-right: 2px;
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
            }
            QTabBar::tab:selected { background: white; color: #111827; font-weight: 600; }
            QTabBar::tab:disabled { color: #94a3b8; background: #eef2f7; }
            QPushButton {
                background: #1f6feb;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 8px 12px;
                font-weight: 600;
            }
            QPushButton:hover { background: #1859bd; }
            QPushButton[secondary="true"] { background: #e9edf3; color: #1b1f24; }
            QPushButton[secondary="true"]:hover { background: #dfe5ec; }
            QPlainTextEdit, QTableWidget, QComboBox, QSpinBox, QDoubleSpinBox {
                background: white;
                color: #111827;
                border: 1px solid #d9dee7;
                border-radius: 8px;
                selection-background-color: #bfdbfe;
                selection-color: #111827;
            }
            QHeaderView::section { background: #334155; color: white; padding: 4px; border: none; }
            QStatusBar { background: #e9edf3; color: #111827; }
            QToolBar { background: #f8fafc; border-bottom: 1px solid #d9dee7; spacing: 6px; }
        """)

    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------
    def _tab_geometry(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        case_box = QGroupBox("Cas 3D")
        case_form = QFormLayout(case_box)
        self.case_combo = QComboBox(); self.case_combo.addItems(["Plaque uniforme", "Plaque avec VBH"])
        self.case_combo.currentIndexChanged.connect(self._update_vbh_enabled)
        self.bc_combo = QComboBox(); self.bc_combo.addItems(["clamped", "free"])
        case_form.addRow("Configuration", self.case_combo)
        case_form.addRow("CL", self.bc_combo)

        plate_box = QGroupBox("Plaque")
        plate_form = QFormLayout(plate_box)
        self.lx_spin = self._dbl(0.50, 0.01, 10.0, 0.01, " m", 4)
        self.ly_spin = self._dbl(0.40, 0.01, 10.0, 0.01, " m", 4)
        self.h_spin = self._dbl(0.002, 0.0001, 1.0, 0.0001, " m", 6)
        plate_form.addRow("Longueur Lx", self.lx_spin)
        plate_form.addRow("Largeur Ly", self.ly_spin)
        plate_form.addRow("Épaisseur h0", self.h_spin)

        mat_box = QGroupBox("Matériau")
        mat_form = QFormLayout(mat_box)
        self.E_spin = self._dbl(69e9, 1e6, 5e11, 1e9, " Pa", 0)
        self.nu_spin = self._dbl(0.33, 0.0, 0.499, 0.01, decimals=4)
        self.rho_spin = self._dbl(2700.0, 1.0, 50000.0, 10.0, " kg/m³", 2)
        self.eta_spin = self._dbl(0.002, 0.0, 1.0, 0.001, decimals=4)
        mat_form.addRow("Module E", self.E_spin)
        mat_form.addRow("Poisson ν", self.nu_spin)
        mat_form.addRow("Densité ρ", self.rho_spin)
        mat_form.addRow("Facteur pertes η", self.eta_spin)

        vbh_box = QGroupBox("Trou noir vibratoire")
        vbh_form = QFormLayout(vbh_box)
        self.cx_spin = self._dbl(0.25, -10.0, 10.0, 0.01, " m", 4)
        self.cy_spin = self._dbl(0.20, -10.0, 10.0, 0.01, " m", 4)
        self.r_vbh_spin = self._dbl(0.06, 0.001, 5.0, 0.001, " m", 4)
        self.rt_spin = self._dbl(0.005, 0.0, 5.0, 0.001, " m", 4)
        self.h_res_spin = self._dbl(0.0003, 0.00001, 1.0, 0.00005, " m", 6)
        self.exp_spin = self._dbl(2.0, 1.0, 10.0, 0.1, decimals=3)
        vbh_form.addRow("Centre x", self.cx_spin)
        vbh_form.addRow("Centre y", self.cy_spin)
        vbh_form.addRow("Rayon VBH", self.r_vbh_spin)
        vbh_form.addRow("Rayon troncature", self.rt_spin)
        vbh_form.addRow("Épaisseur résiduelle", self.h_res_spin)
        vbh_form.addRow("Exposant m", self.exp_spin)

        self.vbh_group = vbh_box
        self._update_vbh_enabled()

        layout.addWidget(case_box)
        layout.addWidget(plate_box)
        layout.addWidget(mat_box)
        layout.addWidget(vbh_box)
        layout.addStretch(1)
        return w

    def _tab_mesh(self):
        w = QWidget()
        layout = QVBoxLayout(w)

        box = QGroupBox("Maillage 3D")
        form = QFormLayout(box)
        self.order_combo = QComboBox(); self.order_combo.addItems(["Ordre 1", "Ordre 2 (Tetra10)"])
        self.order_combo.setCurrentIndex(1)
        self.h_global_spin = self._dbl(0.012, 0.001, 1.0, 0.001, " m", 4)
        self.h_local_spin = self._dbl(0.004, 0.0005, 1.0, 0.0005, " m", 4)
        self.refine_radius_spin = self._dbl(0.030, 0.001, 1.0, 0.001, " m", 4)
        self.transition_spin = self._dbl(0.010, 0.001, 1.0, 0.001, " m", 4)
        self.top_nu_spin = QSpinBox(); self.top_nu_spin.setRange(5, 200); self.top_nu_spin.setValue(17)
        self.top_nv_spin = QSpinBox(); self.top_nv_spin.setRange(5, 200); self.top_nv_spin.setValue(13)
        self.optimize_ho_check = QCheckBox("Optimiser les éléments high-order")
        self.show_edges_check = QCheckBox("Afficher les arêtes")
        self.show_edges_check.setChecked(True)

        form.addRow("Ordre éléments", self.order_combo)
        form.addRow("Taille globale", self.h_global_spin)
        form.addRow("Taille locale", self.h_local_spin)
        form.addRow("Rayon raffinement", self.refine_radius_spin)
        form.addRow("Transition", self.transition_spin)
        form.addRow("BSpline νu", self.top_nu_spin)
        form.addRow("BSpline νv", self.top_nv_spin)
        form.addRow("", self.optimize_ho_check)
        form.addRow("", self.show_edges_check)

        self.mesh_info = QPlainTextEdit(); self.mesh_info.setReadOnly(True)
        self.mesh_info.setPlaceholderText("Les statistiques de maillage apparaîtront ici.")

        layout.addWidget(box)
        layout.addWidget(self.mesh_info)
        layout.addStretch(1)
        return w

    def _tab_modal(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        box = QGroupBox("Analyse modale 3D")
        form = QFormLayout(box)
        self.n_modes_spin = QSpinBox(); self.n_modes_spin.setRange(1, 300); self.n_modes_spin.setValue(20)
        self.mode_combo = QComboBox(); self.mode_combo.currentIndexChanged.connect(self.on_show_mode)
        self.component_combo = QComboBox(); self.component_combo.addItems(["norm", "ux", "uy", "uz"])
        self.scale_spin = self._dbl(1.0, 0.01, 1e6, 0.1, decimals=2)
        self.compare_modal_check = QCheckBox("Comparer uniforme / VBH")
        self.compare_modal_check.setChecked(True)
        self.damping_ratio_spin = self._dbl(0.01, 0.0, 1.0, 0.001, decimals=4)
        form.addRow("Nombre de modes", self.n_modes_spin)
        form.addRow("Mode affiché", self.mode_combo)
        form.addRow("Composante", self.component_combo)
        form.addRow("Facteur d'échelle", self.scale_spin)
        form.addRow("Amortissement ζ", self.damping_ratio_spin)
        form.addRow("", self.compare_modal_check)
        layout.addWidget(box)
        layout.addStretch(1)
        return w

    def _tab_frf(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        box = QGroupBox("Réponse fréquentielle")
        form = QFormLayout(box)
        self.method_combo = QComboBox(); self.method_combo.addItems(["Directe", "Modale"])
        self.fmin_spin = self._dbl(50.0, 0.01, 1e6, 10.0, " Hz", 2)
        self.fmax_spin = self._dbl(700.0, 1.0, 1e6, 10.0, " Hz", 2)
        self.n_freq_spin = QSpinBox(); self.n_freq_spin.setRange(10, 200000); self.n_freq_spin.setValue(300)
        self.frf_modes_spin = QSpinBox(); self.frf_modes_spin.setRange(1, 500); self.frf_modes_spin.setValue(40)
        self.ex_x_spin = self._dbl(0.10, 0.0, 10.0, 0.01, " m", 4)
        self.ex_y_spin = self._dbl(0.10, 0.0, 10.0, 0.01, " m", 4)
        self.ex_z_spin = self._dbl(0.002, 0.0, 10.0, 0.0001, " m", 6)
        self.rx_spin = self._dbl(0.35, 0.0, 10.0, 0.01, " m", 4)
        self.ry_spin = self._dbl(0.25, 0.0, 10.0, 0.01, " m", 4)
        self.rz_spin = self._dbl(0.002, 0.0, 10.0, 0.0001, " m", 6)
        self.dir_combo = QComboBox(); self.dir_combo.addItems(["ux", "uy", "uz"])
        form.addRow("Méthode", self.method_combo)
        form.addRow("f min", self.fmin_spin)
        form.addRow("f max", self.fmax_spin)
        form.addRow("Nb points", self.n_freq_spin)
        form.addRow("Nb modes (si modale)", self.frf_modes_spin)
        form.addRow("Excitation x", self.ex_x_spin)
        form.addRow("Excitation y", self.ex_y_spin)
        form.addRow("Excitation z", self.ex_z_spin)
        form.addRow("Réponse x", self.rx_spin)
        form.addRow("Réponse y", self.ry_spin)
        form.addRow("Réponse z", self.rz_spin)
        form.addRow("Direction mesurée", self.dir_combo)
        layout.addWidget(box)
        layout.addStretch(1)
        return w

    def _tab_post(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        box = QGroupBox("Post-traitement")
        form = QFormLayout(box)
        self.cut_center_check = QCheckBox("Coupe passant par le centre du VBH")
        self.cut_center_check.setChecked(True)
        self.show_profile_check = QCheckBox("Afficher le profil usiné")
        self.show_profile_check.setChecked(True)
        form.addRow("", self.cut_center_check)
        form.addRow("", self.show_profile_check)
        layout.addWidget(box)
        layout.addStretch(1)
        return w

    def _results_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        self.results_table = QTableWidget(0, 2)
        self.results_table.setHorizontalHeaderLabels(["Clé", "Valeur"])
        layout.addWidget(self.results_table)
        return w

    def _plots_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        self.plot_panel = PlotPanel()
        layout.addWidget(self.plot_panel)
        return w

    def _console_tab(self):
        w = QWidget()
        layout = QVBoxLayout(w)
        self.console = QPlainTextEdit(); self.console.setReadOnly(True)
        layout.addWidget(self.console)
        return w

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def _dbl(self, value, vmin=0.0, vmax=1e9, step=0.001, suffix="", decimals=6):
        s = QDoubleSpinBox()
        s.setRange(vmin, vmax)
        s.setDecimals(decimals)
        s.setSingleStep(step)
        s.setValue(value)
        s.setSuffix(suffix)
        return s

    def _update_vbh_enabled(self):
        enabled = self.case_combo.currentText() == "Plaque avec VBH"
        self.vbh_group.setEnabled(enabled)

    def _log(self, text: str):
        self.console.appendPlainText(text)
        self.statusBar().showMessage(text, 4000)

    def _set_results(self, data: dict):
        self.results_table.setRowCount(0)
        for k, v in data.items():
            r = self.results_table.rowCount()
            self.results_table.insertRow(r)
            self.results_table.setItem(r, 0, QTableWidgetItem(str(k)))
            self.results_table.setItem(r, 1, QTableWidgetItem(str(v)))

    def _collect_params(self):
        return {
            "case": "uniform" if self.case_combo.currentIndex() == 0 else "vbh",
            "plate": {
                "lx": self.lx_spin.value(),
                "ly": self.ly_spin.value(),
                "h0": self.h_spin.value(),
                "boundary_condition": self.bc_combo.currentText(),
            },
            "material": {
                "E": self.E_spin.value(),
                "nu": self.nu_spin.value(),
                "rho": self.rho_spin.value(),
                "eta": self.eta_spin.value(),
            },
            "vbh": {
                "cx": self.cx_spin.value(),
                "cy": self.cy_spin.value(),
                "radius": self.r_vbh_spin.value(),
                "truncation_radius": self.rt_spin.value(),
                "h_residual": self.h_res_spin.value(),
                "m": self.exp_spin.value(),
            },
            "mesh": {
                "order": 1 if self.order_combo.currentIndex() == 0 else 2,
                "h_global": self.h_global_spin.value(),
                "h_local": self.h_local_spin.value(),
                "refinement_radius": self.refine_radius_spin.value(),
                "transition_thickness": self.transition_spin.value(),
                "top_surface_nu": self.top_nu_spin.value(),
                "top_surface_nv": self.top_nv_spin.value(),
                "optimize_high_order": self.optimize_ho_check.isChecked(),
                "show_edges": self.show_edges_check.isChecked(),
            },
            "modal": {
                "n_modes": self.n_modes_spin.value(),
                "compare_uniform_vs_vbh": self.compare_modal_check.isChecked(),
                "component": self.component_combo.currentText(),
                "scale": self.scale_spin.value(),
                "damping_ratio": self.damping_ratio_spin.value(),
            },
            "frf": {
                "method": self.method_combo.currentText().lower(),
                "fmin": self.fmin_spin.value(),
                "fmax": self.fmax_spin.value(),
                "n_freq": self.n_freq_spin.value(),
                "n_modes": self.frf_modes_spin.value(),
                "excitation": [self.ex_x_spin.value(), self.ex_y_spin.value(), self.ex_z_spin.value()],
                "response": [self.rx_spin.value(), self.ry_spin.value(), self.rz_spin.value()],
                "direction": self.dir_combo.currentText(),
                "response_type": "displacement",
            },
            "post": {
                "center_cut": self.cut_center_check.isChecked(),
                "show_profile": self.show_profile_check.isChecked(),
            },
        }

    def _load_params(self, cfg: dict):
        self.case_combo.setCurrentIndex(0 if cfg.get("case", "uniform") == "uniform" else 1)
        plate = cfg.get("plate", {})
        self.lx_spin.setValue(plate.get("lx", self.lx_spin.value()))
        self.ly_spin.setValue(plate.get("ly", self.ly_spin.value()))
        self.h_spin.setValue(plate.get("h0", self.h_spin.value()))
        bc = plate.get("boundary_condition", self.bc_combo.currentText())
        idx = max(0, self.bc_combo.findText(bc))
        self.bc_combo.setCurrentIndex(idx)

        material = cfg.get("material", {})
        self.E_spin.setValue(material.get("E", self.E_spin.value()))
        self.nu_spin.setValue(material.get("nu", self.nu_spin.value()))
        self.rho_spin.setValue(material.get("rho", self.rho_spin.value()))
        self.eta_spin.setValue(material.get("eta", self.eta_spin.value()))

        vbh = cfg.get("vbh", {})
        self.cx_spin.setValue(vbh.get("cx", self.cx_spin.value()))
        self.cy_spin.setValue(vbh.get("cy", self.cy_spin.value()))
        self.r_vbh_spin.setValue(vbh.get("radius", self.r_vbh_spin.value()))
        self.rt_spin.setValue(vbh.get("truncation_radius", self.rt_spin.value()))
        self.h_res_spin.setValue(vbh.get("h_residual", self.h_res_spin.value()))
        self.exp_spin.setValue(vbh.get("m", self.exp_spin.value()))

        mesh = cfg.get("mesh", {})
        self.order_combo.setCurrentIndex(0 if mesh.get("order", 2) == 1 else 1)
        self.h_global_spin.setValue(mesh.get("h_global", self.h_global_spin.value()))
        self.h_local_spin.setValue(mesh.get("h_local", self.h_local_spin.value()))
        self.refine_radius_spin.setValue(mesh.get("refinement_radius", self.refine_radius_spin.value()))
        self.transition_spin.setValue(mesh.get("transition_thickness", self.transition_spin.value()))
        self.top_nu_spin.setValue(mesh.get("top_surface_nu", self.top_nu_spin.value()))
        self.top_nv_spin.setValue(mesh.get("top_surface_nv", self.top_nv_spin.value()))
        self.optimize_ho_check.setChecked(mesh.get("optimize_high_order", self.optimize_ho_check.isChecked()))
        self.show_edges_check.setChecked(mesh.get("show_edges", self.show_edges_check.isChecked()))

        modal = cfg.get("modal", {})
        self.n_modes_spin.setValue(modal.get("n_modes", self.n_modes_spin.value()))
        self.compare_modal_check.setChecked(modal.get("compare_uniform_vs_vbh", self.compare_modal_check.isChecked()))
        self.scale_spin.setValue(modal.get("scale", self.scale_spin.value()))
        self.damping_ratio_spin.setValue(modal.get("damping_ratio", self.damping_ratio_spin.value()))
        component = modal.get("component", self.component_combo.currentText())
        idx = max(0, self.component_combo.findText(component))
        self.component_combo.setCurrentIndex(idx)

        frf = cfg.get("frf", {})
        self.method_combo.setCurrentIndex(0 if str(frf.get("method", "directe")).startswith("direct") else 1)
        self.fmin_spin.setValue(frf.get("fmin", self.fmin_spin.value()))
        self.fmax_spin.setValue(frf.get("fmax", self.fmax_spin.value()))
        self.n_freq_spin.setValue(frf.get("n_freq", self.n_freq_spin.value()))
        self.frf_modes_spin.setValue(frf.get("n_modes", self.frf_modes_spin.value()))
        ex = frf.get("excitation", [self.ex_x_spin.value(), self.ex_y_spin.value(), self.ex_z_spin.value()])
        rx = frf.get("response", [self.rx_spin.value(), self.ry_spin.value(), self.rz_spin.value()])
        self.ex_x_spin.setValue(ex[0]); self.ex_y_spin.setValue(ex[1]); self.ex_z_spin.setValue(ex[2])
        self.rx_spin.setValue(rx[0]); self.ry_spin.setValue(rx[1]); self.rz_spin.setValue(rx[2])
        direction = frf.get("direction", self.dir_combo.currentText())
        idx = max(0, self.dir_combo.findText(direction))
        self.dir_combo.setCurrentIndex(idx)

        self._update_vbh_enabled()

    # --------------------------------------------------------
    # Actions
    # --------------------------------------------------------
    def on_preview_mesh(self):
        params = self._collect_params()
        self._log("Aperçu 3D en cours...")
        try:
            result = self.manager.preview_mesh(params)
            self.mesh_result = result
            mesh = result.get("mesh")
            stats = result.get("stats", {})
            self.viewer.show_mesh(mesh, show_edges=self.show_edges_check.isChecked())
            self.mesh_info.setPlainText(json.dumps(stats, indent=2, ensure_ascii=False))
            self._set_results({"action": "preview_mesh", **stats})
            self._log("Aperçu 3D terminé.")
        except Exception as exc:
            self._log(f"[ERREUR] preview_mesh: {exc}")
            QMessageBox.critical(self, "Erreur aperçu", str(exc))

    def on_generate_mesh(self):
        params = self._collect_params()
        self._log("Génération du maillage 3D...")
        try:
            result = self.manager.generate_mesh(params)
            self.mesh_result = result
            mesh = result.get("mesh")
            stats = result.get("stats", {})
            self.viewer.show_mesh(mesh, show_edges=self.show_edges_check.isChecked())
            self.mesh_info.setPlainText(json.dumps(stats, indent=2, ensure_ascii=False))
            self._set_results({"action": "generate_mesh", **stats})
            self._log("Maillage 3D généré.")
        except Exception as exc:
            self._log(f"[ERREUR] generate_mesh: {exc}")
            QMessageBox.critical(self, "Erreur maillage", str(exc))

    def on_compute_modal(self):
        params = self._collect_params()
        self._log("Calcul modal 3D...")
        try:
            result = self.manager.solve_modal(params)
            self.modal_result = result
            freqs = np.asarray(result.get("freqs", []), dtype=float)
            self.mode_combo.clear()
            for i, f in enumerate(freqs, start=1):
                self.mode_combo.addItem(f"Mode {i} - {f:.3f} Hz", i - 1)
            self.plot_panel.plot_modes(freqs)
            summary = {
                "action": "modal",
                "nb_modes": len(freqs),
                "f1_Hz": float(freqs[0]) if len(freqs) else "-",
            }
            comparison = result.get("comparison")
            if comparison is not None:
                summary["f1_uniform_Hz"] = float(comparison["frequencies_uniform_hz"][0])
                summary["f1_vbh_Hz"] = float(comparison["frequencies_abh_hz"][0])
            self._set_results(summary)
            self._log("Calcul modal terminé.")
            if len(freqs):
                self.on_show_mode()
        except Exception as exc:
            self._log(f"[ERREUR] modal: {exc}")
            QMessageBox.critical(self, "Erreur calcul modal", str(exc))

    def on_show_mode(self):
        if self.modal_result is None:
            return
        mode_index = self.mode_combo.currentData()
        if mode_index is None:
            return
        component = self.component_combo.currentText()
        scale = self.scale_spin.value()
        self._log(f"Affichage mode {mode_index + 1} ({component})...")
        try:
            result = self.manager.get_mode_shape(self.modal_result, mode_index, component, scale)
            mesh = result.get("mesh")
            self.viewer.show_mode(mesh, scalars_name=result.get("scalars_name", "mode_shape"), show_edges=False)
        except Exception as exc:
            self._log(f"[ERREUR] show_mode: {exc}")

    def on_compute_frf(self):
        params = self._collect_params()
        self._log("Calcul FRF 3D...")
        try:
            result = self.manager.compute_frf(params)
            self.frf_result = result
            freq = np.asarray(result.get("freq", []), dtype=float)
            H = np.asarray(result.get("H", []))
            if len(freq) and len(H):
                self.plot_panel.plot_curve(freq, np.abs(H), title="FRF 3D", xlabel="Fréquence (Hz)", ylabel="|H|")
            self._set_results({
                "action": "frf",
                "méthode": self.method_combo.currentText(),
                "fmin": self.fmin_spin.value(),
                "fmax": self.fmax_spin.value(),
                "n_points": self.n_freq_spin.value(),
            })
            self._log("Calcul FRF terminé.")
        except Exception as exc:
            self._log(f"[ERREUR] frf: {exc}")
            QMessageBox.critical(self, "Erreur calcul FRF", str(exc))

    def on_save_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "Sauver la configuration", "config_3d.json", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._collect_params(), f, indent=2, ensure_ascii=False)
            self._log(f"Configuration sauvée: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Erreur sauvegarde", str(exc))

    def on_load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Charger une configuration", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            self._load_params(cfg)
            self._log(f"Configuration chargée: {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Erreur chargement", str(exc))


if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    win = MainWindow3D()
    win.show()
    sys.exit(app.exec())
