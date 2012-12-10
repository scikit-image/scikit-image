import warnings

import numpy as np

try:
    import matplotlib as mpl
    from matplotlib.figure import Figure
    from matplotlib import _pylab_helpers
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.backends.backend_qt4 import FigureManagerQT
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg
except ImportError:
    FigureCanvasQTAgg = object  # hack to prevent nosetest and autodoc errors
    LinearSegmentedColormap = object
    print("Could not import matplotlib -- skimage.viewer not available.")

try:
    from PyQt4 import QtGui
except ImportError:
    print("Could not import PyQt4 -- skimage.viewer not available.")


__all__ = ['init_qtapp', 'start_qtapp', 'RequiredAttr', 'figimage',
           'LinearColormap', 'ClearColormap', 'FigureCanvas', 'new_plot']


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


class LinearColormap(LinearSegmentedColormap):
    """LinearSegmentedColormap in which color varies smoothly.

    This class is a simplification of LinearSegmentedColormap, which doesn't
    support jumps in color intensities.

    Parameters
    ----------
    name : str
        Name of colormap.

    segmented_data : dict
        Dictionary of 'red', 'green', 'blue', and (optionally) 'alpha' values.
        Each color key contains a list of `x`, `y` tuples. `x` must increase
        monotonically from 0 to 1 and corresponds to input values for a
        mappable object (e.g. an image). `y` corresponds to the color
        intensity.

    """
    def __init__(self, name, segmented_data, **kwargs):
        segmented_data = dict((key, [(x, y, y) for x, y in value])
                              for key, value in segmented_data.iteritems())
        LinearSegmentedColormap.__init__(self, name, segmented_data, **kwargs)


class ClearColormap(LinearColormap):
    """Color map that varies linearly from alpha = 0 to 1
    """
    def __init__(self, rgb, max_alpha=1, name='clear_color'):
        r, g, b = rgb
        cg_speq = {'blue':  [(0.0, b), (1.0, b)],
                   'green': [(0.0, g), (1.0, g)],
                   'red':   [(0.0, r), (1.0, r)],
                   'alpha': [(0.0, 0.0), (1.0, max_alpha)]}
        LinearColormap.__init__(self, name, cg_speq)


class FigureCanvas(FigureCanvasQTAgg):
    """Canvas for displaying images."""
    def __init__(self, figure, **kwargs):
        self.fig = figure
        FigureCanvasQTAgg.__init__(self, self.fig)
        FigureCanvasQTAgg.setSizePolicy(self,
                                        QtGui.QSizePolicy.Expanding,
                                        QtGui.QSizePolicy.Expanding)
        FigureCanvasQTAgg.updateGeometry(self)

    #TODO: Consider overriding Matplotlib key-event handling
    # def keyPressEvent(self, event):
        # # Override key events defined by Matplotlib
        # event.ignore()

    # def keyReleaseEvent(self, event):
        # # Override key events defined by Matplotlib
        # event.ignore()


def new_canvas(*args, **kwargs):
    """Return a new figure canvas."""
    allnums = _pylab_helpers.Gcf.figs.keys()
    num = max(allnums) + 1 if allnums else 1

    FigureClass = kwargs.pop('FigureClass', Figure)
    figure = FigureClass(*args, **kwargs)
    canvas = FigureCanvas(figure)
    fig_manager = FigureManagerQT(canvas, num)
    return fig_manager.canvas


def new_plot(parent=None, subplot_kw=None, **fig_kw):
    """Return new figure and axes.

    Parameters
    ----------
    parent : QtWidget
        Qt widget that displays the plot objects. If None, you must manually
        call ``canvas.setParent`` and pass the parent widget.
    subplot_kw : dict
        Keyword arguments passed ``matplotlib.figure.Figure.add_subplot``.
    fig_kw : dict
        Keyword arguments passed ``matplotlib.figure.Figure``.
    """
    if subplot_kw is None:
        subplot_kw = {}
    canvas = new_canvas(**fig_kw)
    canvas.setParent(parent)

    fig = canvas.figure
    ax = fig.add_subplot(1, 1, 1, **subplot_kw)
    return fig, ax


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
    dpi = dpi if dpi is not None else mpl.rcParams['figure.dpi']
    kwargs.setdefault('interpolation', 'nearest')
    kwargs.setdefault('cmap', 'gray')

    h, w, d = np.atleast_3d(image).shape
    figsize = np.array((w, h), dtype=float) / dpi * scale

    fig, ax = new_plot(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    ax.set_axis_off()
    ax.imshow(image, **kwargs)
    return fig, ax
