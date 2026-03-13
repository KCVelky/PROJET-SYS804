# ui/main_window.py

from __future__ import annotations

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTabWidget,
    QMessageBox,
)

from core import SimulationManager
from ui.tabs import PlateTab, BlackHoleTab, ExcitationTab, MeshTab, AnalysisTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.manager = SimulationManager(load_defaults=True)

        self.setWindowTitle("Étude EF d'une plaque avec trou noir vibratoire")
        self.resize(1500, 900)

        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)

        button_row = QHBoxLayout()
        self.btn_apply = QPushButton("Appliquer tous les paramètres")
        self.btn_reload_defaults = QPushButton("Recharger les valeurs par défaut")

        button_row.addWidget(self.btn_apply)
        button_row.addWidget(self.btn_reload_defaults)
        button_row.addStretch(1)

        self.tabs = QTabWidget()

        self.tab_plate = PlateTab(self.manager)
        self.tab_black_hole = BlackHoleTab(self.manager)
        self.tab_excitation = ExcitationTab(self.manager)
        self.tab_mesh = MeshTab(self.manager)
        self.tab_analysis = AnalysisTab(self.manager, self.apply_all_parameters)

        self.tabs.addTab(self.tab_plate, "Plaque")
        self.tabs.addTab(self.tab_black_hole, "Trou noir")
        self.tabs.addTab(self.tab_excitation, "Capteurs / force")
        self.tabs.addTab(self.tab_mesh, "Maillage")
        self.tabs.addTab(self.tab_analysis, "Lancer analyse")

        root.addLayout(button_row)
        root.addWidget(self.tabs)

        self.btn_apply.clicked.connect(self.apply_all_parameters)
        self.btn_reload_defaults.clicked.connect(self.reload_defaults)

        self.statusBar().showMessage("Interface prête.")

    def apply_all_parameters(self) -> bool:
        try:
            self.tab_plate.apply_to_manager()
            self.tab_black_hole.apply_to_manager()
            self.tab_excitation.apply_to_manager()
            self.tab_mesh.apply_to_manager()
            self.tab_analysis.refresh_sensor_list()

            self.statusBar().showMessage("Paramètres appliqués avec succès.")
            return True

        except Exception as exc:
            QMessageBox.critical(self, "Erreur de paramètres", str(exc))
            self.statusBar().showMessage("Échec d'application des paramètres.")
            return False

    def reload_defaults(self) -> None:
        try:
            self.manager.load_default_case()
            self.tab_plate.load_from_manager()
            self.tab_black_hole.load_from_manager()
            self.tab_excitation.load_from_manager()
            self.tab_mesh.load_from_manager()
            self.tab_analysis.refresh_sensor_list()
            self.statusBar().showMessage("Valeurs par défaut rechargées.")
        except Exception as exc:
            QMessageBox.critical(self, "Erreur", str(exc))
            self.statusBar().showMessage("Erreur lors du rechargement des valeurs.")