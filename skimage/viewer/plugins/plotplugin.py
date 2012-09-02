import numpy as np
from PyQt4 import QtGui

import matplotlib.pyplot as plt

from ..utils import MatplotlibCanvas
from .base import Plugin


class PlotCanvas(MatplotlibCanvas):
    """Canvas for displaying images.

    This canvas derives from Matplotlib, and has attributes `fig` and `ax`,
    which point to Matplotlib figure and axes.
    """
    def __init__(self, parent, height, width, **kwargs):
        self.fig, self.ax = plt.subplots(figsize=(height, width), **kwargs)
        super(PlotCanvas, self).__init__(parent, self.fig, **kwargs)
        self.setMinimumHeight(150)

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
        self.canvas = PlotCanvas(self, height, width)
        self.fig = self.canvas.fig
        #TODO: Converted color is slightly different than Qt background.
        qpalette = QtGui.QPalette()
        qcolor = qpalette.color(QtGui.QPalette.Window)
        bgcolor = qcolor.toRgb().value()
        if np.isscalar(bgcolor):
            bgcolor = str(bgcolor / 255.)
        self.fig.patch.set_facecolor(bgcolor)
        self.ax = self.canvas.ax
        self.layout.addWidget(self.canvas, self.row, 0)
