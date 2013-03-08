import numpy as np

try:
    from matplotlib import lines
except ImportError:
    print("Could not import matplotlib -- skimage.viewer not available.")


__all__ = ['CanvasToolBase', 'ToolHandles']


def _pass(*args):
    pass


class CanvasToolBase(object):
    """Base canvas tool for matplotlib axes.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Matplotlib axes where tool is displayed.
    on_move : function
        Function called whenever a control handle is moved.
        This function must accept the end points of line as the only argument.
    on_release : function
        Function called whenever the control handle is released.
    on_enter : function
        Function called whenever the "enter" key is pressed.
    useblit : bool
        If True, update canvas by blitting, which is much faster than normal
        redrawing (turn off for debugging purposes).
    """
    def __init__(self, ax, on_move=None, on_enter=None, on_release=None,
                 useblit=True):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.img_background = None
        self.cids = []
        self._artists = []
        self.active = True

        if useblit:
            self.connect_event('draw_event', self._blit_on_draw_event)
        self.useblit = useblit

        self.callback_on_move = _pass if on_move is None else on_move
        self.callback_on_enter = _pass if on_enter is None else on_enter
        self.callback_on_release = _pass if on_release is None else on_release

        self.connect_event('key_press_event', self._on_key_press)

    def connect_event(self, event, callback):
        """Connect callback with an event.

        This should be used in lieu of `figure.canvas.mpl_connect` since this
        function stores call back ids for later clean up.
        """
        cid = self.canvas.mpl_connect(event, callback)
        self.cids.append(cid)

    def disconnect_events(self):
        """Disconnect all events created by this widget."""
        for c in self.cids:
            self.canvas.mpl_disconnect(c)

    def ignore(self, event):
        """Return True if event should be ignored.

        This method (or a version of it) should be called at the beginning
        of any event callback.
        """
        return not self.active

    def set_visible(self, val):
        for artist in self._artists:
            artist.set_visible(val)

    def _blit_on_draw_event(self, event=None):
        self.img_background = self.canvas.copy_from_bbox(self.ax.bbox)
        self._draw_artists()

    def _draw_artists(self):
        for artist in self._artists:
            self.ax.draw_artist(artist)

    def remove(self):
        """Remove artists and events from axes.

        Note that the naming here mimics the interface of Matplotlib artists.
        """
        #TODO: For some reason, RectangleTool doesn't get properly removed
        self.disconnect_events()
        for a in self._artists:
            a.remove()

    def redraw(self):
        """Redraw image and canvas artists.

        This method should be called by subclasses when artists are updated.
        """
        if self.useblit and self.img_background is not None:
            self.canvas.restore_region(self.img_background)
            self._draw_artists()
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def _on_key_press(self, event):
        if event.key == 'enter':
            self.callback_on_enter(self.geometry)
            self.set_visible(False)
            self.redraw()

    @property
    def geometry(self):
        """Geometry information that gets passed to callback functions."""
        raise NotImplementedError


class ToolHandles(object):
    """Control handles for canvas tools.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Matplotlib axes where tool handles are displayed.
    x, y : 1D arrays
        Coordinates of control handles.
    marker : str
        Shape of marker used to display handle. See `matplotlib.pyplot.plot`.
    marker_props : dict
        Additional marker properties. See :class:`matplotlib.lines.Line2D`.
    """
    def __init__(self, ax, x, y, marker='o', marker_props=None):
        self.ax = ax

        props = dict(marker=marker, markersize=7, mfc='w', ls='none',
                     alpha=0.5, visible=False)
        props.update(marker_props if marker_props is not None else {})
        self._markers = lines.Line2D(x, y, animated=True, **props)
        self.ax.add_line(self._markers)
        self.artist = self._markers

    @property
    def x(self):
        return self._markers.get_xdata()

    @property
    def y(self):
        return self._markers.get_ydata()

    def set_data(self, pts, y=None):
        """Set x and y positions of handles"""
        if y is not None:
            x = pts
            pts = np.array([x, y])
        self._markers.set_data(pts)

    def set_visible(self, val):
        self._markers.set_visible(val)

    def set_animated(self, val):
        self._markers.set_animated(val)

    def draw(self):
        self.ax.draw_artist(self._markers)

    def closest(self, x, y):
        """Return index and pixel distance to closest index."""
        pts = np.transpose((self.x, self.y))
        # Transform data coordinates to pixel coordinates.
        pts = self.ax.transData.transform(pts)
        diff = pts - ((x, y))
        dist = np.sqrt(np.sum(diff**2, axis=1))
        return np.argmin(dist), np.min(dist)
