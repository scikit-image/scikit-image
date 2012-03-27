import numpy as np
import scipy.ndimage as ndi
from skimage.util.dtype import dtype_range

from .base import Plugin


__all__ = ['LineProfile']


class LineProfile(Plugin):
    """Plugin to compute interpolated intensity under a scan line on an image.

    Parameters
    ----------
    image_window : ImageViewer instance.
        Window containing image used in measurement.
    useblit : bool
        If True, use blitting to speed up animation. Only available on some
        backends. If None, set to True when using Agg backend, otherwise False.
    linewidth : float
        Line width for interpolation. Wider lines average over more pixels.
    epsilon : float
        Maximum pixel distance allowed when selecting end point of scan line.
    limits : tuple or {None, 'image', 'dtype'}
        (minimum, maximum) intensity limits for plotted profile. The following
        special values are defined:

            None : rescale based on min/max intensity along selected scan line.
            'image' : fixed scale based on min/max intensity in image.
            'dtype' : fixed scale based on min/max intensity of image dtype.
    """


    def __init__(self, image_window, useblit=None,
                 linewidth=1, epsilon=5, limits='image'):

        Plugin.__init__(self, image_window, useblit=useblit, figsize=(8, 3))

        self.linewidth = linewidth
        self.epsilon = epsilon

        if limits == 'image':
            self.limits = (np.min(self.image), np.max(self.image))
        elif limits == 'dtype':
            self.limits = dtype_range[self.image.dtype.type]
        elif limits is None or len(limits) == 2:
            self.limits = limits
        else:
            raise ValueError("Unrecognized `limits`: %s" % limits)

        if not limits is None:
            self.ax.set_ylim(self.limits)

        h, w = self.image.shape

        self._init_end_pts = np.array([[w/3, h/2], [2*w/3, h/2]])
        self.end_pts = self._init_end_pts.copy()

        x, y = np.transpose(self.end_pts)
        self.scan_line = self.imgview.ax.plot(x, y, 'y-s', markersize=5,
                                             lw=linewidth, alpha=0.5,
                                             solid_capstyle='butt')[0]
        self.artists.append(self.scan_line)

        scan_data = profile_line(self.image, self.end_pts)
        self.profile = self.ax.plot(scan_data, 'k-')[0]
        self._autoscale_view()

        self._active_pt = None

        self.connect_event('key_press_event', self.on_key_press)
        self.connect_event('button_press_event', self.on_mouse_press)
        self.connect_event('button_release_event', self.on_mouse_release)
        self.connect_event('motion_notify_event', self.on_move)
        self.connect_event('scroll_event', self.on_scroll)

        self.imgview.redraw()
        print self.help()

    def remove_artist(self, artist):
        """Disconnect all artists created by this widget."""
        #self.ax.remove(artist)
        self.ax.lines = []


    def help(self):
        helpstr = ("Line profile tool",
                   "+ and - keys or mouse scroll changes width of scan line.",
                   "Select and drag ends of the scan line to adjust it.")
        return '\n'.join(helpstr)

    def get_profile(self):
        """Return intensity profile of the selected line.

        Returns
        -------
        end_pts: (2, 2) array
            The positions ((x1, y1), (x2, y2)) of the line ends.
        profile: 1d array
            Profile of intensity values.
        """
        end_pts = self.scan_line.get_xydata()
        profile = self.profile.get_ydata()
        return end_pts, profile

    def on_scroll(self, event):
        if not event.inaxes: return
        if event.button == 'up':
            self._thicken_scan_line()
        elif event.button == 'down':
            self._shrink_scan_line()

    def on_key_press(self, event):
        if not event.inaxes: return
        elif event.key == '+':
            self._thicken_scan_line()
        elif event.key == '-':
            self._shrink_scan_line()
        elif event.key == 'r':
            self.reset()

    def _thicken_scan_line(self):
        self.linewidth += 1
        self.line_changed(None, None)

    def _shrink_scan_line(self):
        if self.linewidth > 1:
            self.linewidth -= 1
            self.line_changed(None, None)

    def _autoscale_view(self):
        if self.limits is None:
            self.ax.autoscale_view(tight=True)
        else:
            self.ax.autoscale_view(scaley=False, tight=True)

    def get_pt_under_cursor(self, event):
        """Return index of the end point under cursor, if sufficiently close"""
        xy = np.asarray(self.scan_line.get_xydata())
        xyt = self.scan_line.get_transform().transform(xy)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.sqrt((xt - event.x)**2 + (yt - event.y)**2)
        indseq = np.nonzero(np.equal(d, np.amin(d)))[0]
        ind = indseq[0]
        if d[ind] >= self.epsilon:
            ind = None
        return ind

    def on_mouse_press(self, event):
        if event.button != 1: return
        if event.inaxes==None: return
        self._active_pt = self.get_pt_under_cursor(event)

    def on_mouse_release(self, event):
        if event.button != 1: return
        self._active_pt = None

    def on_move(self, event):
        if event.button != 1: return
        if self._active_pt is None: return
        if not self.imgview.ax.in_axes(event): return
        x,y = event.xdata, event.ydata
        self.line_changed(x, y)

    def reset(self):
        self.end_pts = self._init_end_pts.copy()
        self.scan_line.set_data(np.transpose(self.end_pts))
        self.line_changed(None, None)

    def line_changed(self, x, y):
        if x is not None:
            self.end_pts[self._active_pt, :] = x, y
        self.scan_line.set_data(np.transpose(self.end_pts))
        self.scan_line.set_linewidth(self.linewidth)

        scan = profile_line(self.image, self.end_pts, linewidth=self.linewidth)
        self.profile.set_xdata(np.arange(scan.shape[0]))
        self.profile.set_ydata(scan)

        self.ax.relim()

        if self.useblit:
            self.imgview.canvas.restore_region(self.img_background)
            self.ax.draw_artist(self.scan_line)
            self.ax.draw_artist(self.profile)
            self.imgview.canvas.blit(self.imgview.ax.bbox)

        self._autoscale_view()

        self.imgview.redraw()
        self.redraw()


def profile_line(img, end_pts, linewidth=1):
    """Return the intensity profile of an image measured along a scan line.

    Parameters
    ----------
    img : 2d array
        The image.
    end_pts: (2, 2) list
        End points ((x1, y1), (x2, y2)) of scan line.
    linewidth: int
        Width of the scan, perpendicular to the line

    Returns
    -------
    return_value : array
        The intensity profile along the scan line. The length of the profile
        is the ceil of the computed length of the scan line.
    """
    point1, point2 = end_pts
    x1, y1 = point1 = np.asarray(point1, dtype = float)
    x2, y2 = point2 = np.asarray(point2, dtype = float)
    dx, dy = point2 - point1

    # Quick calculation if perfectly horizontal or vertical (remove?)
    if x1 == x2:
        pixels = img[min(y1, y2) : max(y1, y2)+1,
                     x1 - linewidth / 2 :  x1 + linewidth / 2 + 1]
        intensities  = pixels.mean(axis = 1)
        return intensities
    elif y1 == y2:
        pixels = img[y1 - linewidth / 2 :  y1 + linewidth / 2 + 1,
                     min(x1, x2) : max(x1, x2)+1]
        intensities = pixels.mean(axis = 0)
        return intensities

    theta = np.arctan2(dy,dx)
    a = dy/dx
    b = y1 - a * x1
    length = np.hypot(dx, dy)

    line_x = np.linspace(min(x1, x2), max(x1, x2), np.ceil(length))
    line_y = line_x * a + b
    y_width = abs(linewidth * np.cos(theta)/2)
    perp_ys = np.array([np.linspace(yi - y_width,
                                    yi + y_width, linewidth) for yi in line_y])
    perp_xs = - a * perp_ys + (line_x +  a * line_y)[:, np.newaxis]

    perp_lines = np.array([perp_ys, perp_xs])
    pixels = ndi.map_coordinates(img, perp_lines)
    intensities = pixels.mean(axis=1)

    return intensities

