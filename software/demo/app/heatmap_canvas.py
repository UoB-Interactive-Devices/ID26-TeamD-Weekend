import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


class HeatmapCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.figure = Figure(figsize=(7.0, 3.2), dpi=100)
        self.figure.set_facecolor("#f7fbff")

        self.ax_front = self.figure.add_subplot(1, 2, 1)
        self.ax_back = self.figure.add_subplot(1, 2, 2)

        self.front_image = self.ax_front.imshow(
            np.zeros((7, 7)),
            cmap="viridis",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        self.back_image = self.ax_back.imshow(
            np.zeros((6, 7)),
            cmap="viridis",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )

        self.ax_front.set_title("Front Matrix (7x7)")
        self.ax_back.set_title("Back Matrix (6x7)")

        for axis in (self.ax_front, self.ax_back):
            axis.set_xticks([])
            axis.set_yticks([])

        self.figure.tight_layout()
        super().__init__(self.figure)

    def update_matrices(self, front_matrix: np.ndarray, back_matrix: np.ndarray):
        self.front_image.set_array(front_matrix)
        self.back_image.set_array(back_matrix)
        self.draw_idle()
