from PyQt4 import QtGui

import matplotlib as mpl
from ..widgets import Slider


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

    def __init__(self, image_viewer, image_filter=None, height=100, width=400,
                 useblit=None):
        self.image_viewer = image_viewer
        QtGui.QDialog.__init__(self, image_viewer)
        self.image_viewer.plugins.append(self)

        self.setWindowTitle(self.name)
        self.layout = QtGui.QGridLayout(self)
        self.resize(width, height)
        self.row = 0
        if image_filter is not None:
            self.image_filter = image_filter

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

    def filter_image(self, *args):
        arguments = [self._get_value(a) for a in self.arguments]
        kwargs = dict([(name, self._get_value(a))
                       for name, a in self.keyword_arguments.iteritems()])
        self.image_filter(*arguments, **kwargs)

    def _get_value(self, param):
        if hasattr(param, 'val'):
            return param.val()
        else:
            return param

    def add_widget(self, widget):
        if widget.ptype == 'kwarg':
            name = widget.name.replace(' ', '_')
            self.keyword_arguments[name] = widget
            widget.callback = self.filter_image
        elif widget.ptype == 'arg':
            self.arguments.append(widget)
            widget.callback = self.filter_image
        self.layout.addWidget(widget, self.row, 0)
        self.row += 1

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

        Parameters
        ----------
        event : str
            Matplotlib event.
        callback : function
            Callback function with a matplotlib Event object as its argument.
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
