import numpy as np

try:
    from matplotlib import lines
except ImportError:
    print("Could not import matplotlib -- skimage.viewer not available.")

from base import CanvasToolBase, ToolHandles


__all__ = ['LineTool', 'ThickLineTool']


class LineTool(CanvasToolBase):
    """Widget for line selection in a plot.

    Parameters
    ----------
    on_move : function
        Function called whenever a control handle is moved.
        This function must accept the end points of line as the only argument.
    on_release : function
        Function called whenever the control handle is released.
    on_enter : function
        Function called whenever the "enter" key is pressed.

    Attributes
    ----------
    end_pts : 2D array
        End points of line ((x1, y1), (x2, y2)).
    """
    def __init__(self, ax, x, y, on_move=None, on_release=None, on_enter=None,
                 maxdist=10, lineprops=None):
        super(LineTool, self).__init__(ax, on_move=on_move, on_enter=on_enter,
                                       on_release=on_release)

        props = dict(color='r', linewidth=1, alpha=0.4, solid_capstyle='butt')
        props.update(lineprops if lineprops is not None else {})
        self.linewidth = props['linewidth']
        self.maxdist = maxdist
        self._active_pt = None

        self.end_pts = np.transpose([x, y])

        self._line = lines.Line2D(x, y, animated=True, **props)
        ax.add_line(self._line)

        self._handles = ToolHandles(ax, x, y)
        self._handles.set_visible(True)
        self._artists = [self._line, self._handles.artist]

        if on_enter is None:
            def on_enter(pts):
                x, y = np.transpose(pts)
                print "length = %0.2f" % np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        self.callback_on_enter = on_enter

        self.connect_event('button_press_event', self.on_mouse_press)
        self.connect_event('button_release_event', self.on_mouse_release)
        self.connect_event('motion_notify_event', self.on_move)

    def on_mouse_press(self, event):
        if event.button != 1 and event.inaxes == None:
            return
        self.set_visible(True)
        idx, px_dist = self._handles.closest(event.x, event.y)
        if px_dist < self.maxdist:
            self._active_pt = idx
        else:
            self._active_pt = 0
            x, y = event.xdata, event.ydata
            self.end_pts = np.array([[x, y], [x, y]])

    def on_mouse_release(self, event):
        if event.button != 1:
            return
        self._active_pt = None
        self.callback_on_release(self.geometry)

    def on_move(self, event):
        if event.button != 1 or self._active_pt is None:
            return
        if not self.ax.in_axes(event):
            return
        self.update(event.xdata, event.ydata)
        self.callback_on_move(self.geometry)

    def update(self, x, y):
        if x is not None:
            self.end_pts[self._active_pt, :] = x, y
        self._line.set_data(np.transpose(self.end_pts))
        self._handles.set_data(np.transpose(self.end_pts))
        self._line.set_linewidth(self.linewidth)

        self.redraw()

    @property
    def geometry(self):
        return self.end_pts


class ThickLineTool(LineTool):

    def __init__(self, ax, x, y, on_move=None, on_enter=None, on_release=None,
                 maxdist=10, lineprops=None):
        super(ThickLineTool, self).__init__(ax, x, y,
                                            on_move=on_move,
                                            on_enter=on_enter,
                                            on_release=on_release,
                                            maxdist=maxdist,
                                            lineprops=lineprops)

        self.connect_event('scroll_event', self.on_scroll)
        self.connect_event('key_press_event', self.on_key_press)

    def on_scroll(self, event):
        if not event.inaxes:
            return
        if event.button == 'up':
            self._thicken_scan_line()
        elif event.button == 'down':
            self._shrink_scan_line()

    def on_key_press(self, event):
        if event.key == '+':
            self._thicken_scan_line()
        elif event.key == '-':
            self._shrink_scan_line()

    def _thicken_scan_line(self):
        self.linewidth += 1
        self.update(None, None)

    def _shrink_scan_line(self):
        if self.linewidth > 1:
            self.linewidth -= 1
            self.update(None, None)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from skimage import data

    image = data.camera()

    f, ax = plt.subplots()
    ax.imshow(image, interpolation='nearest')
    h, w = image.shape

    # line_tool = LineTool(ax, [w/3, 2*w/3], [h/2, h/2])
    line_tool = ThickLineTool(ax, [w/3, 2*w/3], [h/2, h/2])
    plt.show()
