from PyQt4 import QtGui

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
from skimage.io._plugins.q_color_mixer import IntelligentSlider

from skimage.viewer.utils import clear_red


class PlotCanvas(FigureCanvasQTAgg):
    """Canvas for displaying images.

    This canvas derives from Matplotlib, and has attributes `fig` and `ax`,
    which point to Matplotlib figure and axes.
    """
    def __init__(self, parent, height, width, **kwargs):
        print height, width
        self.fig, self.ax = plt.subplots(figsize=(height, width), **kwargs)

        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self,
                                        QtGui.QSizePolicy.Expanding,
                                        QtGui.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        # Note: `setParent` must be called after `FigureCanvasQTAgg.__init__`.
        self.setParent(parent)


class Plugin(QtGui.QDialog):
    """Base class for widgets that interact with the axes.

    Parameters
    ----------
    image_viewer : ImageViewer instance.
        Window containing image used in measurement/manipulation.
    callback : function
        Function that gets called to update ImageViewer. Alternatively, this
        can also be defined as a method in a Plugin subclass.
    height, width : int
        Size of plugin window in pixels.
    useblit : bool
        If True, use blitting to speed up animation. Only available on some
        backends. If None, set to True when using Agg backend, otherwise False.

    Attributes
    ----------
    image_viewer : ImageViewer
        Window containing image used in measurement.
    image : array
        Image used in measurement/manipulation.
    """
    name = 'Plugin'
    draws_on_image = False

    def __init__(self, image_viewer, callback=None, height=100, width=400,
                 useblit=None):
        self.image_viewer = image_viewer
        QtGui.QDialog.__init__(self, image_viewer)
        self.image_viewer.plugins.append(self)

        self.setWindowTitle(self.name)
        self.layout = QtGui.QGridLayout(self)
        self.resize(width, height)
        self.row = 0
        if callback is not None:
            self.callback = callback

        self.arguments = [image_viewer.original_image]
        self.keyword_arguments= {}

        self.image = self.image_viewer.image

        if useblit is None:
            useblit = True if mpl.backends.backend.endswith('Agg') else False
        self.useblit = useblit
        self.cids = []
        self.artists = []

        if self.draws_on_image:
            self.connect_event('draw_event', self.on_draw)

    def on_draw(self, event):
        """Save image background when blitting.

        The saved image is used to "clear" the figure before redrawing artists.
        """
        if self.useblit:
            bbox = self.image_viewer.ax.bbox
            self.img_background = self.image_viewer.canvas.copy_from_bbox(bbox)

    def caller(self, *args):
        arguments = [self._get_value(a) for a in self.arguments]
        kwargs = dict([(name, self._get_value(a))
                       for name, a in self.keyword_arguments.iteritems()])
        self.callback(*arguments, **kwargs)

    def _get_value(self, param):
        if hasattr(param, 'val'):
            return param.val()
        else:
            return param

    def add_argument(self, name, low, high, callback, **kwargs):
        name, slider = self.add_slider(name, low, high, callback, **kwargs)
        self.arguments[name] = slider

    def add_keyword_argument(self, name, low, high, callback, **kwargs):
        name, slider = self.add_slider(name, low, high, callback, **kwargs)
        self.keyword_arguments[name] = slider

    def add_slider(self, name, low, high, callback, **kwargs):
        slider = IntelligentSlider(name, low, high, callback,
                                   orientation='horizontal', **kwargs)
        self.layout.addWidget(slider, self.row, 0)
        self.row += 1
        return name.replace(' ', '_'), slider

    def closeEvent(self, event):
        """Disconnect all artists and events from ImageViewer.

        Note that events must be connected using `self.connect_event` and
        artists must be appended to `self.artists`.
        """
        self.disconnect_image_events()
        self.remove_artists()
        self.image_viewer.plugins.remove(self)
        self.image_viewer.redraw()
        self.close()

    def connect_event(self, event, callback):
        """Connect callback with an event.

        This should be used in lieu of `figure.canvas.mpl_connect` since this
        function stores call back ids for later clean up.
        """
        cid = self.image_viewer.connect_event(event, callback)
        self.cids.append(cid)

    def disconnect_image_events(self):
        """Disconnect all events created by this widget."""
        for c in self.cids:
            self.image_viewer.disconnect_event(c)

    def remove_artists(self):
        """Disconnect artists that are connected to the *image plot*."""
        for a in self.artists:
            self.image_viewer.remove_artist(a)


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
    def __init__(self, image_viewer, **kwargs):
        Plugin.__init__(self, image_viewer, **kwargs)
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


class OverlayPlugin(Plugin):
    """Plugin for ImageViewer that displays an overlay on top of main image.

    Attributes
    ----------
    overlay : array
        Overlay displayed on top of image. This overlay defaults to a color map
        with alpha values varying linearly from 0 to 1.
    """

    def __init__(self, image_viewer, **kwargs):
        Plugin.__init__(self, image_viewer, **kwargs)
        self.overlay_cmap = clear_red
        self._overlay_plot = None
        self._overlay = None

    @property
    def overlay(self):
        return self._overlay

    @overlay.setter
    def overlay(self, image):
        self._overlay = image
        ax = self.image_viewer.ax
        if image is None:
            ax.images.remove(self._overlay_plot)
            self._overlay_plot = None
        elif self._overlay_plot is None:
            self._overlay_plot = ax.imshow(image, cmap=self.overlay_cmap)
        else:
            self._overlay_plot.set_array(image)
        self.image_viewer.redraw()
