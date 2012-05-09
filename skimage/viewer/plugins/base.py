import matplotlib as mpl
import matplotlib.pyplot as plt

from .. import utils


__all__ = ['Plugin']


class Plugin(object):
    """Base class for widgets that interact with the axes.

    Parameters
    ----------
    image_viewer : ImageViewer instance.
        Window containing image used in measurement/manipulation.
    useblit : bool
        If True, use blitting to speed up animation. Only available on some
        backends. If None, set to True when using Agg backend, otherwise False.
    figure : :class:`~matplotlib.figure.Figure`
        If None, create a figure with a single axes.
    no_toolbar : bool
        If True, figure created by plugin has no toolbar. This has no effect
        on figures passed into `Plugin`.

    Attributes
    ----------
    ax : :class:`~matplotlib.axes.Axes`
        The parent axes for the widget. Note, if `figure` is passed as a
        parameter, `ax` is set to None since there could be multiple axes
        attached to the figure.
    figure : :class:`~matplotlib.figure.Figure`
        The parent figure for the widget.
    canvas : :class:`~matplotlib.backend_bases.FigureCanvasBase` subclass
        The parent figure canvs for the widget.
    imgview : ImageViewer
        Window containing image used in measurement.
    image : array
        Image used in measurement/manipulation.
    active : bool
        If False, the widget does not respond to events.
    """

    def __init__(self, image_viewer, useblit=None, figsize=None, figure=None,
                 no_toolbar=True):
        self.imgview = image_viewer
        self.image = self.imgview._img
        # Add Plugin to imgview's list to prevent garbage-collection.
        # Reference must be removed when closing plugin.
        self.imgview.plugins.append(self)

        if figure is None:
            if figsize is None:
                figsize = plt.rcParams['figure.figsize']
            with utils.toolbar_off(no_toolbar):
                figure = plt.figure(figsize=figsize)
            self.figure = figure
            self.canvas = figure.canvas
            self.ax = figure.add_subplot(111)
        else:
            self.figure = figure
            self.canvas = figure.canvas
            self.ax = None

        if useblit is None:
            useblit = True if mpl.backends.backend.endswith('Agg') else False
        self.useblit = useblit

        self.active = True
        self.cids = []
        self.artists = []

        self.connect_event('draw_event', self.on_draw)
        self.canvas.mpl_connect('close_event', self.on_close)

    def redraw(self):
        self.canvas.draw_idle()

    def on_draw(self, event):
        """Save image background when blitting.

        The saved image is used to "clear" the figure before redrawing artists.
        """
        if self.useblit:
            bbox = self.imgview.ax.bbox
            self.img_background = self.imgview.canvas.copy_from_bbox(bbox)

    def on_close(self, event):
        """Disconnect all artists and events from ImageViewer.

        Note that events must be connected using `self.connect_event` and
        artists must be appended to `self.artists`.
        """
        self.disconnect_image_events()
        self.remove_artists()
        self.imgview.plugins.remove(self)
        self.imgview.redraw()

    def ignore(self, event):
        """Return True if event should be ignored.

        This method (or a version of it) should be called at the beginning
        of any event callback.
        """
        return not self.active

    def connect_event(self, event, callback):
        """Connect callback with an event.

        This should be used in lieu of `figure.canvas.mpl_connect` since this
        function stores call back ids for later clean up.
        """
        cid = self.imgview.connect_event(event, callback)
        self.cids.append(cid)

    def disconnect_image_events(self):
        """Disconnect all events created by this widget."""
        for c in self.cids:
            self.imgview.disconnect_event(c)

    def remove_artists(self):
        """Disconnect artists that are connected to the *image plot*."""
        for a in self.artists:
            self.imgview.remove_artist(a)

