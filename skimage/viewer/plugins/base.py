from PyQt4 import QtGui
from PyQt4.QtCore import Qt

import matplotlib as mpl


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
    """
    name = 'Plugin'
    draws_on_image = False

    def __init__(self, image_filter=None, height=0, width=400, useblit=None):
        super(Plugin, self).__init__()

        self.image_viewer = None
        self.image_filter = image_filter

        self.setWindowTitle(self.name)
        self.layout = QtGui.QGridLayout(self)
        self.resize(width, height)
        self.row = 0

        self.arguments = []
        self.keyword_arguments= {}

        if useblit is None:
            useblit = True if mpl.backends.backend.endswith('Agg') else False
        self.useblit = useblit
        self.cids = []
        self.artists = []

    @property
    def image_viewer(self):
        if self._image_viewer is None:
            raise RuntimeError("Plugin is not attached to ImageViewer")
        return self._image_viewer

    @image_viewer.setter
    def image_viewer(self, image_viewer):
        self._image_viewer = image_viewer

    def attach(self, image_viewer):
        self.setParent(image_viewer)
        self.setWindowFlags(Qt.Dialog)

        self.image_viewer = image_viewer
        self.image_viewer.plugins.append(self)
        #TODO: Always passing image as first argument may be bad assumption.
        self.arguments.append(self.image_viewer.original_image)

        if self.draws_on_image:
            self.connect_image_event('draw_event', self.on_draw)
        # Call filter so that filtered image matches widget values
        self.filter_image()

    def on_draw(self, event):
        """Save image background when blitting.

        The saved image is used to "clear" the figure before redrawing artists.
        """
        if self.useblit:
            bbox = self.image_viewer.ax.bbox
            self.img_background = self.image_viewer.canvas.copy_from_bbox(bbox)

    def filter_image(self, *args):
        if self.image_filter is None:
            return
        arguments = [self._get_value(a) for a in self.arguments]
        kwargs = dict([(name, self._get_value(a))
                       for name, a in self.keyword_arguments.iteritems()])
        filtered = self.image_filter(*arguments, **kwargs)
        self.display_filtered_image(filtered)

    def display_filtered_image(self, image):
        """Override this method to display image on image viewer."""
        self.image_viewer.image = image

    def _get_value(self, param):
        return param if not hasattr(param, 'val') else param.val()

    def add_widget(self, widget):
        if widget.ptype == 'kwarg':
            name = widget.name.replace(' ', '_')
            self.keyword_arguments[name] = widget
            widget.callback = self.filter_image
        elif widget.ptype == 'arg':
            self.arguments.append(widget)
            widget.callback = self.filter_image
        elif widget.ptype == 'plugin':
            widget.callback = self.update_plugin
        self.layout.addWidget(widget, self.row, 0)
        self.row += 1

    def __iadd__(self, widget):
        self.add_widget(widget)
        return self

    def update_plugin(self, name, value):
        setattr(self, name, value)

    def closeEvent(self, event):
        """Disconnect all artists and events from ImageViewer.

        Note that events must be connected using `self.connect_image_event` and
        artists must be appended to `self.artists`.
        """
        self.disconnect_image_events()
        self.remove_image_artists()
        self.image_viewer.plugins.remove(self)
        self.image_viewer.redraw()
        self.close()

    def connect_image_event(self, event, callback):
        """Connect callback with an event in the image viewer.

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

    def remove_image_artists(self):
        """Disconnect artists that are connected to the image viewer."""
        for a in self.artists:
            self.image_viewer.remove_artist(a)
