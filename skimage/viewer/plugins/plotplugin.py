import numpy as np
from PyQt4 import QtGui

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg

from .base import Plugin


class PlotCanvas(FigureCanvasQTAgg):
    """Canvas for displaying images.

    This canvas derives from Matplotlib, and has attributes `fig` and `ax`,
    which point to Matplotlib figure and axes.
    """
    def __init__(self, parent, height, width, **kwargs):
        self.fig, self.ax = plt.subplots(figsize=(height, width), **kwargs)

        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self,
                                        QtGui.QSizePolicy.Expanding,
                                        QtGui.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        # Note: `setParent` must be called after `FigureCanvasQTAgg.__init__`.
        self.setParent(parent)
        self.setMinimumHeight(150)


class PlotPlugin(Plugin):
    """Plugin for ImageViewer that contains a plot Canvas.

    Parameters
    ----------
    image_viewer : ImageViewer instance.
        Window containing image used in measurement/manipulation.

    Attributes
    ----------
    image_viewer : ImageViewer
        Window containing image used in measurement.
    image : array
        Image used in measurement/manipulation.
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
