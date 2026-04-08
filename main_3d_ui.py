from __future__ import annotations

import sys
from PySide6.QtWidgets import QApplication

from ui.main_window_3d import MainWindow3D

import core.simulation_manager_3d as sm3d
print(sm3d.__file__)

import core.simulation_manager_3d as sm3d
print("SimulationManager3D chargé depuis :", sm3d.__file__)

from core.simulation_manager_3d import SimulationManager3D
print("preview_mesh existe ?", hasattr(SimulationManager3D, "preview_mesh"))
print("generate_mesh existe ?", hasattr(SimulationManager3D, "generate_mesh"))
print("solve_modal existe ?", hasattr(SimulationManager3D, "solve_modal"))

def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow3D()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
