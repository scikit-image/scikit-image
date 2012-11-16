import numpy as np
import matplotlib as mpl
from matplotlib import lines


__all__ = ['CanvasToolBase', 'ToolHandles']


class CanvasToolBase(object):
    """Base canvas tool for matplotlib axes.

    Parameters
    ----------
    """
    def __init__(self, ax, useblit=None):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.cids = []
        self._artists = []
        self.active = True

        if useblit is None:
            useblit = True if mpl.backends.backend.endswith('Agg') else False
        self.useblit = useblit
        if useblit:
            self.canvas.draw()
            self.img_background = self.canvas.copy_from_bbox(self.ax.bbox)

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
        for a in self._artists:
            a.set_visible(val)


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
    marker_props : see :class:`matplotlib.lines.Line2D`.
    """
    def __init__(self, ax, x, y, marker='o', markerprops=None):
        self.ax = ax

        props = dict(mfc='w', ls='none', alpha=0.5, visible=False)
        props.update(markerprops if markerprops is not None else {})
        self._markers = lines.Line2D(x, y, marker=marker, **props)
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
