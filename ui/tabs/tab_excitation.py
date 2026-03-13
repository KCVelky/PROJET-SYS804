# ui/tabs/tab_excitation.py

from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGroupBox,
    QFormLayout,
    QDoubleSpinBox,
    QSpinBox,
    QComboBox,
    QLineEdit,
)

from core import SimulationManager


class ExcitationTab(QWidget):
    def __init__(self, manager: SimulationManager) -> None:
        super().__init__()
        self.manager = manager

        layout = QVBoxLayout(self)

        force_group = QGroupBox("Force ponctuelle harmonique")
        force_form = QFormLayout(force_group)

        self.force_x_mm = self._make_double_box(0.0, 5000.0, 100.0, 3)
        self.force_y_mm = self._make_double_box(0.0, 5000.0, 100.0, 3)
        self.force_amp = self._make_double_box(0.001, 1e6, 1.0, 4)
        self.freq_start = self._make_double_box(0.1, 1e5, 50.0, 3)
        self.freq_end = self._make_double_box(0.1, 1e5, 700.0, 3)
        self.n_points = QSpinBox()
        self.n_points.setRange(10, 20000)
        self.n_points.setValue(400)
        self.phase_deg = self._make_double_box(-360.0, 360.0, 0.0, 3)

        self.direction_combo = QComboBox()
        self.direction_combo.addItems(["w"])

        force_form.addRow("Position x [mm]", self.force_x_mm)
        force_form.addRow("Position y [mm]", self.force_y_mm)
        force_form.addRow("Amplitude [N]", self.force_amp)
        force_form.addRow("Fréquence début [Hz]", self.freq_start)
        force_form.addRow("Fréquence fin [Hz]", self.freq_end)
        force_form.addRow("Nombre de points", self.n_points)
        force_form.addRow("Phase [deg]", self.phase_deg)
        force_form.addRow("Direction", self.direction_combo)

        sensor_group = QGroupBox("Capteur principal")
        sensor_form = QFormLayout(sensor_group)

        self.sensor_name = QLineEdit()
        self.sensor_x_mm = self._make_double_box(0.0, 5000.0, 350.0, 3)
        self.sensor_y_mm = self._make_double_box(0.0, 5000.0, 250.0, 3)

        self.sensor_response_combo = QComboBox()
        self.sensor_response_combo.addItems(["displacement", "velocity", "acceleration"])

        sensor_form.addRow("Nom du capteur", self.sensor_name)
        sensor_form.addRow("Position x [mm]", self.sensor_x_mm)
        sensor_form.addRow("Position y [mm]", self.sensor_y_mm)
        sensor_form.addRow("Type de réponse", self.sensor_response_combo)

        layout.addWidget(force_group)
        layout.addWidget(sensor_group)
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
        exc = summary["excitation"]
        sensors = summary["sensors"]

        self.force_x_mm.setValue(exc["x"] * 1e3)
        self.force_y_mm.setValue(exc["y"] * 1e3)
        self.force_amp.setValue(exc["amplitude"])
        self.freq_start.setValue(exc["frequency_start"])
        self.freq_end.setValue(exc["frequency_end"])
        self.n_points.setValue(exc["n_points"])
        self.phase_deg.setValue(exc["phase_deg"])
        self.direction_combo.setCurrentText(exc["direction"])

        if sensors:
            first_name = next(iter(sensors))
            sensor = sensors[first_name]
            self.sensor_name.setText(sensor["name"])
            self.sensor_x_mm.setValue(sensor["x"] * 1e3)
            self.sensor_y_mm.setValue(sensor["y"] * 1e3)
            self.sensor_response_combo.setCurrentText(sensor["response_type"])
        else:
            self.sensor_name.setText("S1")

    def apply_to_manager(self) -> None:
        self.manager.set_excitation(
            x=self.force_x_mm.value() / 1e3,
            y=self.force_y_mm.value() / 1e3,
            amplitude=self.force_amp.value(),
            frequency_start=self.freq_start.value(),
            frequency_end=self.freq_end.value(),
            n_points=self.n_points.value(),
            phase_deg=self.phase_deg.value(),
            direction=self.direction_combo.currentText(),
        )

        self.manager.clear_sensors()
        self.manager.set_sensor(
            name=self.sensor_name.text().strip() or "S1",
            x=self.sensor_x_mm.value() / 1e3,
            y=self.sensor_y_mm.value() / 1e3,
            response_type=self.sensor_response_combo.currentText(),
        )