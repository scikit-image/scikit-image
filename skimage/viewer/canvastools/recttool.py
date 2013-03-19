try:
    from matplotlib.widgets import RectangleSelector
except ImportError:
    RectangleSelector = object
    print("Could not import matplotlib -- skimage.viewer not available.")

from skimage.viewer.canvastools.base import CanvasToolBase
from skimage.viewer.canvastools.base import ToolHandles


__all__ = ['RectangleTool']


class RectangleTool(CanvasToolBase, RectangleSelector):
    """Widget for selecting a rectangular region in a plot.

    After making the desired selection, press "Enter" to accept the selection
    and call the `on_enter` callback function.

    Parameters
    ----------
    ax : :class:`matplotlib.axes.Axes`
        Matplotlib axes where tool is displayed.
    on_move : function
        Function called whenever a control handle is moved.
        This function must accept the rectangle extents as the only argument.
    on_release : function
        Function called whenever the control handle is released.
    on_enter : function
        Function called whenever the "enter" key is pressed.
    maxdist : float
        Maximum pixel distance allowed when selecting control handle.
    rect_props : dict
        Properties for :class:`matplotlib.patches.Rectangle`. This class
        redefines defaults in :class:`matplotlib.widgets.RectangleSelector`.

    Attributes
    ----------
    extents : tuple
        Rectangle extents: (xmin, xmax, ymin, ymax).
    """

    def __init__(self, ax, on_move=None, on_release=None, on_enter=None,
                 maxdist=10, rect_props=None):
        CanvasToolBase.__init__(self, ax, on_move=on_move,
                                on_enter=on_enter, on_release=on_release)

        props = dict(edgecolor=None, facecolor='r', alpha=0.15)
        props.update(rect_props if rect_props is not None else {})
        if props['edgecolor'] is None:
            props['edgecolor'] = props['facecolor']
        RectangleSelector.__init__(self, ax, lambda *args: None,
                                   rectprops=props,
                                   useblit=self.useblit)
        # Alias rectangle attribute, which is initialized in RectangleSelector.
        self._rect = self.to_draw
        self._rect.set_animated(True)

        self.maxdist = maxdist
        self.active_handle = None
        self._extents_on_press = None

        if on_enter is None:
            def on_enter(extents):
                print("(xmin=%.3g, xmax=%.3g, ymin=%.3g, ymax=%.3g)" % extents)
        self.callback_on_enter = on_enter

        props = dict(mec=props['edgecolor'])
        self._corner_order = ['NW', 'NE', 'SE', 'SW']
        xc, yc = self.corners
        self._corner_handles = ToolHandles(ax, xc, yc, marker_props=props)

        self._edge_order = ['W', 'N', 'E', 'S']
        xe, ye = self.edge_centers
        self._edge_handles = ToolHandles(ax, xe, ye, marker='s',
                                         marker_props=props)

        self._artists = [self._rect,
                         self._corner_handles.artist,
                         self._edge_handles.artist]

    @property
    def _rect_bbox(self):
        x0 = self._rect.get_x()
        y0 = self._rect.get_y()
        width = self._rect.get_width()
        height = self._rect.get_height()
        return x0, y0, width, height

    @property
    def corners(self):
        """Corners of rectangle from lower left, moving clockwise."""
        x0, y0, width, height = self._rect_bbox
        xc = x0, x0 + width, x0 + width, x0
        yc = y0, y0, y0 + height, y0 + height
        return xc, yc

    @property
    def edge_centers(self):
        """Midpoint of rectangle edges from left, moving clockwise."""
        x0, y0, width, height = self._rect_bbox
        w = width / 2.
        h = height / 2.
        xe = x0, x0 + w, x0 + width, x0 + w
        ye = y0 + h, y0, y0 + h, y0 + height
        return xe, ye

    @property
    def extents(self):
        """Return (xmin, xmax, ymin, ymax)."""
        x0, y0, width, height = self._rect_bbox
        xmin, xmax = sorted([x0, x0 + width])
        ymin, ymax = sorted([y0, y0 + height])
        return xmin, xmax, ymin, ymax

    @extents.setter
    def extents(self, extents):
        x1, x2, y1, y2 = extents
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        # Update displayed rectangle
        self._rect.set_x(xmin)
        self._rect.set_y(ymin)
        self._rect.set_width(xmax - xmin)
        self._rect.set_height(ymax - ymin)
        # Update displayed handles
        self._corner_handles.set_data(*self.corners)
        self._edge_handles.set_data(*self.edge_centers)

        self.set_visible(True)
        self.redraw()

    def release(self, event):
        if event.button != 1:
            return
        if not self.ax.in_axes(event):
            self.eventpress = None
            return
        RectangleSelector.release(self, event)
        self._extents_on_press = None
        # Undo hiding of rectangle and redraw.
        self.set_visible(True)
        self.redraw()
        self.callback_on_release(self.geometry)

    def press(self, event):
        if event.button != 1 or not self.ax.in_axes(event):
            return
        self._set_active_handle(event)
        if self.active_handle is None:
            # Clear previous rectangle before drawing new rectangle.
            self.set_visible(False)
            self.redraw()
        self.set_visible(True)
        RectangleSelector.press(self, event)

    def _set_active_handle(self, event):
        """Set active handle based on the location of the mouse event"""
        # Note: event.xdata/ydata in data coordinates, event.x/y in pixels
        c_idx, c_dist = self._corner_handles.closest(event.x, event.y)
        e_idx, e_dist = self._edge_handles.closest(event.x, event.y)

        # Set active handle as closest handle, if mouse click is close enough.
        if c_dist > self.maxdist and e_dist > self.maxdist:
            self.active_handle = None
            return
        elif c_dist < e_dist:
            self.active_handle = self._corner_order[c_idx]
        else:
            self.active_handle = self._edge_order[e_idx]

        # Save coordinates of rectangle at the start of handle movement.
        x1, x2, y1, y2 = self.extents
        # Switch variables so that only x2 and/or y2 are updated on move.
        if self.active_handle in ['W', 'SW', 'NW']:
            x1, x2 = x2, event.xdata
        if self.active_handle in ['N', 'NW', 'NE']:
            y1, y2 = y2, event.ydata
        self._extents_on_press = x1, x2, y1, y2

    def onmove(self, event):
        if self.eventpress is None or not self.ax.in_axes(event):
            return

        if self.active_handle is None:
            # New rectangle
            x1 = self.eventpress.xdata
            y1 = self.eventpress.ydata
            x2, y2 = event.xdata, event.ydata
        else:
            x1, x2, y1, y2 = self._extents_on_press
            if self.active_handle in ['E', 'W'] + self._corner_order:
                x2 = event.xdata
            if self.active_handle in ['N', 'S'] + self._corner_order:
                y2 = event.ydata
        self.extents = (x1, x2, y1, y2)
        self.callback_on_move(self.geometry)

    @property
    def geometry(self):
        return self.extents


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage import data

    f, ax = plt.subplots()
    ax.imshow(data.camera(), interpolation='nearest')

    rect_tool = RectangleTool(ax)
    plt.show()
    print("Final selection:")
    rect_tool.callback_on_enter(rect_tool.extents)
