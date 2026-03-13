# ui/tabs/tab_black_hole.py

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QFormLayout,
    QDoubleSpinBox,
    QCheckBox,
)

from core import SimulationManager


class BlackHoleTab(QWidget):
    def __init__(self, manager: SimulationManager) -> None:
        super().__init__()
        self.manager = manager

        layout = QVBoxLayout(self)

        group = QGroupBox("Trou noir vibratoire")
        form = QFormLayout(group)

        self.enabled_checkbox = QCheckBox("Activer le trou noir vibratoire")
        self.xc_mm = self._make_double_box(0.0, 5000.0, 250.0, 3)
        self.yc_mm = self._make_double_box(0.0, 5000.0, 200.0, 3)
        self.radius_mm = self._make_double_box(0.1, 1000.0, 60.0, 3)
        self.truncation_mm = self._make_double_box(0.0, 1000.0, 5.0, 3)
        self.residual_thickness_mm = self._make_double_box(0.001, 100.0, 0.3, 4)
        self.exponent = self._make_double_box(0.1, 10.0, 2.0, 3)

        form.addRow(self.enabled_checkbox)
        form.addRow("Centre xc [mm]", self.xc_mm)
        form.addRow("Centre yc [mm]", self.yc_mm)
        form.addRow("Rayon extérieur [mm]", self.radius_mm)
        form.addRow("Rayon de troncature [mm]", self.truncation_mm)
        form.addRow("Épaisseur résiduelle [mm]", self.residual_thickness_mm)
        form.addRow("Exposant du profil", self.exponent)

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
        bh = summary["black_hole"]

        self.enabled_checkbox.setChecked(bh["enabled"])
        self.xc_mm.setValue(bh["xc"] * 1e3)
        self.yc_mm.setValue(bh["yc"] * 1e3)
        self.radius_mm.setValue(bh["radius"] * 1e3)
        self.truncation_mm.setValue(bh["truncation_radius"] * 1e3)
        self.residual_thickness_mm.setValue(bh["residual_thickness"] * 1e3)
        self.exponent.setValue(bh["exponent"])

    def apply_to_manager(self) -> None:
        self.manager.set_black_hole(
            xc=self.xc_mm.value() / 1e3,
            yc=self.yc_mm.value() / 1e3,
            radius=self.radius_mm.value() / 1e3,
            truncation_radius=self.truncation_mm.value() / 1e3,
            residual_thickness=self.residual_thickness_mm.value() / 1e3,
            exponent=self.exponent.value(),
            enabled=self.enabled_checkbox.isChecked(),
        )