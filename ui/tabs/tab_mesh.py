# ui/tabs/tab_mesh.py

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QFormLayout,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
)

from core import SimulationManager


class MeshTab(QWidget):
    def __init__(self, manager: SimulationManager) -> None:
        super().__init__()
        self.manager = manager

        layout = QVBoxLayout(self)

        group = QGroupBox("Paramètres de maillage")
        form = QFormLayout(group)

        self.element_size_mm = self._make_double_box(0.1, 1000.0, 20.0, 3)
        self.refine_checkbox = QCheckBox("Activer le raffinement local")
        self.refinement_radius_mm = self._make_double_box(0.0, 1000.0, 80.0, 3)
        self.refinement_element_size_mm = self._make_double_box(0.1, 1000.0, 5.0, 3)

        self.element_type_combo = QComboBox()
        self.element_type_combo.addItems(["tri3"])

        form.addRow("Taille globale [mm]", self.element_size_mm)
        form.addRow(self.refine_checkbox)
        form.addRow("Rayon de raffinement [mm]", self.refinement_radius_mm)
        form.addRow("Taille locale [mm]", self.refinement_element_size_mm)
        form.addRow("Type d'élément", self.element_type_combo)

        layout.addWidget(group)
        layout.addStretch(1)

        self.load_from_manager()

    @staticmethod
    def _make_double_box(vmin: float, vmax: float, value: float, decimals: int) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(vmin, vmax)
        box.setDecimals(decimals)
        box.setValue(value)
        return box

    def load_from_manager(self) -> None:
        summary = self.manager.get_case_summary()
        mesh = summary["mesh"]

        self.element_size_mm.setValue(mesh["element_size"] * 1e3)
        self.refine_checkbox.setChecked(mesh["refine_near_black_hole"])
        self.refinement_radius_mm.setValue(mesh["refinement_radius"] * 1e3)
        self.refinement_element_size_mm.setValue(mesh["refinement_element_size"] * 1e3)
        self.element_type_combo.setCurrentText(mesh["element_type"])

    def apply_to_manager(self) -> None:
        self.manager.set_mesh(
            element_size=self.element_size_mm.value() / 1e3,
            refine_near_black_hole=self.refine_checkbox.isChecked(),
            refinement_radius=self.refinement_radius_mm.value() / 1e3,
            refinement_element_size=self.refinement_element_size_mm.value() / 1e3,
            element_type=self.element_type_combo.currentText(),
        )