import numpy as np

try:
    from PyQt4 import QtGui
except ImportError:
    print("Could not import PyQt4 -- skimage.viewer not available.")

from ..utils import new_plot
from .base import Plugin


__all__ = ['PlotPlugin']


class PlotPlugin(Plugin):
    """Plugin for ImageViewer that contains a plot canvas.

    Base class for plugins that contain a Matplotlib plot canvas, which can,
    for example, display an image histogram.

    See base Plugin class for additional details.
    """

    def attach(self, image_viewer):
        super(PlotPlugin, self).attach(image_viewer)
        # Add plot for displaying intensity profile.
        self.add_plot()

    def redraw(self):
        """Redraw plot."""
        self.canvas.draw_idle()

    def add_plot(self, height=4, width=4):
        self.fig, self.ax = new_plot(figsize=(height, width))
        self.canvas = self.fig.canvas
        self.canvas.setMinimumHeight(150)
        #TODO: Converted color is slightly different than Qt background.
        qpalette = QtGui.QPalette()
        qcolor = qpalette.color(QtGui.QPalette.Window)
        bgcolor = qcolor.toRgb().value()
        if np.isscalar(bgcolor):
            bgcolor = str(bgcolor / 255.)
        self.fig.patch.set_facecolor(bgcolor)
        self.layout.addWidget(self.canvas, self.row, 0)
