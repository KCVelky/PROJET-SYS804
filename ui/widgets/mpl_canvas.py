# ui/widgets/mpl_canvas.py

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self) -> None:
        self.figure = Figure(figsize=(7, 5), tight_layout=True)
        super().__init__(self.figure)

    def clear(self) -> None:
        self.figure.clear()
        self.draw_idle()