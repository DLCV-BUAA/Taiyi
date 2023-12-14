from .base_figure import *

class LineFigure(Figure):
    def _plot(self, ax=None):
        for x, y, label in zip(self.x_data, self.y_data, self.legend):
            ax.plot(x, y, label=label)

            
   