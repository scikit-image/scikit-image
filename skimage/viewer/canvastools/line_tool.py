import numpy as np

try:
    from matplotlib import lines
except ImportError:
    print("Could not import matplotlib -- skimage.viewer not available.")

from base import CanvasToolBase, ToolHandles


__all__ = ['LineTool', 'ThickLineTool']


class LineTool(CanvasToolBase):
    """
    Parameters
    ----------
    on_update : function
        Function accepting end points of line as the only argument.

    Attributes
    ----------
    end_pts : 2D array
        End points of line ((x1, y1), (x2, y2)).
    """
    def __init__(self, ax, x, y, on_update=None, on_enter=None, maxdist=10,
                 lineprops=None):
        super(LineTool, self).__init__(ax, on_update=on_update,
                                       on_enter=on_enter)

        props = dict(color='r', linewidth=1, alpha=0.4, solid_capstyle='butt')
        props.update(lineprops if lineprops is not None else {})
        self.linewidth = props['linewidth']
        self.maxdist = maxdist
        self._active_pt = None

        self._init_end_pts = np.transpose([x, y])
        self.end_pts = self._init_end_pts.copy()

        self._line = lines.Line2D(x, y, animated=True, **props)
        ax.add_line(self._line)

        self._handles = ToolHandles(ax, x, y)
        self._handles.set_visible(True)
        self._artists = [self._line, self._handles.artist]

        self.connect_event('button_press_event', self.on_mouse_press)
        self.connect_event('button_release_event', self.on_mouse_release)
        self.connect_event('motion_notify_event', self.on_move)
        self.connect_event('key_press_event', self.on_key_press)

    def on_mouse_press(self, event):
        if event.button != 1:
            return
        if event.inaxes == None:
            return
        idx, px_dist = self._handles.closest(event.x, event.y)
        if px_dist < self.maxdist:
            self._active_pt = idx

    def on_mouse_release(self, event):
        if event.button != 1:
            return
        self._active_pt = None

    def on_move(self, event):
        if event.button != 1:
            return
        if self._active_pt is None:
            return
        if not self.ax.in_axes(event):
            return
        x, y = event.xdata, event.ydata
        self.update(x, y)

    def on_key_press(self, event):
        if event.key == 'enter':
            self.on_enter(self.end_pts)
            self.set_visible(False)
            self.redraw()

    def reset(self):
        self.end_pts = self._init_end_pts.copy()
        self._line.set_data(np.transpose(self.end_pts))
        self._handles.set_data(np.transpose(self.end_pts))
        self.update(None, None)

    def update(self, x, y):
        if x is not None:
            self.end_pts[self._active_pt, :] = x, y
        self._line.set_data(np.transpose(self.end_pts))
        self._handles.set_data(np.transpose(self.end_pts))
        self._line.set_linewidth(self.linewidth)

        self.ax.relim()
        self.redraw()

        self.on_update(self.end_pts)


class ThickLineTool(LineTool):

    def __init__(self, ax, x, y, on_update=None, on_enter=None, maxdist=10,
                 lineprops=None):
        super(ThickLineTool, self).__init__(ax, x, y, on_update=on_update,
                                            on_enter=on_enter, maxdist=maxdist,
                                            lineprops=lineprops)

        self.connect_event('scroll_event', self.on_scroll)

    def on_scroll(self, event):
        if not event.inaxes:
            return
        if event.button == 'up':
            self._thicken_scan_line()
        elif event.button == 'down':
            self._shrink_scan_line()

    def on_key_press(self, event):
        super(ThickLineTool, self).on_key_press(event)

        if event.key == '+':
            self._thicken_scan_line()
        elif event.key == '-':
            self._shrink_scan_line()
        elif event.key == 'r':
            self.reset()

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

    def printer(pts):
        x, y = np.transpose(pts)
        print "length = %0.2f" % np.sqrt(np.diff(x)**2 + np.diff(y)**2)

    # line_tool = LineTool(ax, [w/3, 2*w/3], [h/2, h/2])
    line_tool = ThickLineTool(ax, [w/3, 2*w/3], [h/2, h/2], on_enter=printer)
    plt.show()
