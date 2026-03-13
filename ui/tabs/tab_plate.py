# ui/tabs/tab_plate.py

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QFormLayout,
    QDoubleSpinBox,
    QComboBox,
    QLineEdit,
)

from core import SimulationManager


class PlateTab(QWidget):
    def __init__(self, manager: SimulationManager) -> None:
        super().__init__()
        self.manager = manager

        layout = QVBoxLayout(self)

        plate_group = QGroupBox("Plaque")
        plate_form = QFormLayout(plate_group)

        self.plate_name_edit = QLineEdit()
        self.length_x_mm = self._make_double_box(1.0, 5000.0, 500.0, 3)
        self.length_y_mm = self._make_double_box(1.0, 5000.0, 400.0, 3)
        self.thickness_mm = self._make_double_box(0.01, 100.0, 2.0, 4)

        self.bc_combo = QComboBox()
        self.bc_combo.addItems(["simply_supported", "clamped", "free"])

        plate_form.addRow("Nom de la plaque", self.plate_name_edit)
        plate_form.addRow("Longueur x [mm]", self.length_x_mm)
        plate_form.addRow("Largeur y [mm]", self.length_y_mm)
        plate_form.addRow("Épaisseur [mm]", self.thickness_mm)
        plate_form.addRow("Condition limite", self.bc_combo)

        material_group = QGroupBox("Matériau")
        material_form = QFormLayout(material_group)

        self.material_name_edit = QLineEdit()
        self.young_gpa = self._make_double_box(0.1, 5000.0, 69.0, 4)
        self.poisson = self._make_double_box(0.0, 0.499, 0.33, 4)
        self.density = self._make_double_box(1.0, 50000.0, 2700.0, 3)

        material_form.addRow("Nom du matériau", self.material_name_edit)
        material_form.addRow("Module d'Young [GPa]", self.young_gpa)
        material_form.addRow("Poisson [-]", self.poisson)
        material_form.addRow("Masse volumique [kg/m³]", self.density)

        layout.addWidget(plate_group)
        layout.addWidget(material_group)
        layout.addStretch(1)

        self.load_from_manager()

    @staticmethod
    def _make_double_box(vmin: float, vmax: float, value: float, decimals: int) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(vmin, vmax)
        box.setDecimals(decimals)
        box.setValue(value)
        box.setSingleStep(max((vmax - vmin) / 1000.0, 0.001))
        return box

    def load_from_manager(self) -> None:
        summary = self.manager.get_case_summary()
        plate = summary["plate"]
        material = summary["material"]

        self.plate_name_edit.setText(plate["name"])
        self.length_x_mm.setValue(plate["length_x"] * 1e3)
        self.length_y_mm.setValue(plate["length_y"] * 1e3)
        self.thickness_mm.setValue(plate["thickness"] * 1e3)
        self.bc_combo.setCurrentText(plate["boundary_condition"])

        self.material_name_edit.setText(material["name"])
        self.young_gpa.setValue(material["young_modulus"] / 1e9)
        self.poisson.setValue(material["poisson_ratio"])
        self.density.setValue(material["density"])

    def apply_to_manager(self) -> None:
        self.manager.set_material(
            young_modulus=self.young_gpa.value() * 1e9,
            poisson_ratio=self.poisson.value(),
            density=self.density.value(),
            name=self.material_name_edit.text().strip() or "Generic material",
        )

        self.manager.set_plate(
            length_x=self.length_x_mm.value() / 1e3,
            length_y=self.length_y_mm.value() / 1e3,
            thickness=self.thickness_mm.value() / 1e3,
            boundary_condition=self.bc_combo.currentText(),
            name=self.plate_name_edit.text().strip() or "Rectangular plate",
        )