import warnings

import numpy as np


try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
except ImportError:
    FigureCanvasQTAgg = object  # hack to prevent nosetest and autodoc errors
    print("Could not import matplotlib -- skimage.viewer not available.")

try:
    from PyQt4 import QtGui
except ImportError:
    print("Could not import PyQt4 -- skimage.viewer not available.")


__all__ = ['init_qtapp', 'start_qtapp', 'RequiredAttr', 'figimage',
           'MatplotlibCanvas']


QApp = None


def init_qtapp():
    """Initialize QAppliction.

    The QApplication needs to be initialized before creating any QWidgets
    """
    global QApp
    if QApp is None:
        QApp = QtGui.QApplication([])


def start_qtapp():
    """Start Qt mainloop"""
    QApp.exec_()


class RequiredAttr(object):
    """A class attribute that must be set before use."""

    def __init__(self, msg):
        self.msg = msg
        self.val = None

    def __get__(self, obj, objtype):
        if self.val is None:
            warnings.warn(self.msg)
        return self.val

    def __set__(self, obj, val):
        self.val = val


def figimage(image, scale=1, dpi=None, **kwargs):
    """Return figure and axes with figure tightly surrounding image.

    Unlike pyplot.figimage, this actually plots onto an axes object, which
    fills the figure. Plotting the image onto an axes allows for subsequent
    overlays of axes artists.

    Parameters
    ----------
    image : array
        image to plot
    scale : float
        If scale is 1, the figure and axes have the same dimension as the
        image.  Smaller values of `scale` will shrink the figure.
    dpi : int
        Dots per inch for figure. If None, use the default rcParam.
    """
    dpi = dpi if dpi is not None else plt.rcParams['figure.dpi']
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('cmap', 'gray')

    h, w, d = np.atleast_3d(image).shape
    figsize = np.array((w, h), dtype=float) / dpi * scale

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    ax.set_axis_off()
    ax.imshow(image, **kwargs)
    return fig, ax


class MatplotlibCanvas(FigureCanvasQTAgg):
    """Canvas for displaying images."""
    def __init__(self, parent, figure, **kwargs):
        self.fig = figure
        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self,
                                        QtGui.QSizePolicy.Expanding,
                                        QtGui.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)
        # Note: `setParent` must be called after `FigureCanvasQTAgg.__init__`.
        self.setParent(parent)
