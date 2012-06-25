import matplotlib.pyplot as plt
from skimage.util.dtype import dtype_range
from skimage.exposure import histogram
from numpy import linspace, zeros, ones

from .base import Plugin
from ..widgets.slider import Slider
from .. import utils


__all__ = ['ContrastSetter']


class ContrastSetter(Plugin):
    """Plugin to manually adjust the contrast of an image.

    Only linear adjustments are possible. Source image is not modified.

    Parameters
    ----------
    image_viewer : ImageViewer instance.
        Window containing image used in measurement.

    """

    def __init__(self, image_viewer):
        with utils.toolbar_off():
            figure = plt.figure(figsize=(6.5, 2))
        ax_hist = plt.subplot2grid((6, 1), (0, 0), rowspan=4)
        ax_low = plt.subplot2grid((6, 1), (4, 0), rowspan=1)
        ax_high = plt.subplot2grid((6, 1), (5, 0), rowspan=1)
        self.ax_hist = ax_hist

        Plugin.__init__(self, image_viewer, figure=figure)

        hmin, hmax = dtype_range[self.image.dtype.type]
        if hmax > 255:
            bins = int(hmax - hmin)
        else:
            bins = 256
        self.hist, self.bin_centers = histogram(self.image.data, bins)
        self.cmin = self.bin_centers[0]
        self.cmax = self.bin_centers[-1]

        # draw marker lines before histogram so they're behind histogram
        self.low_marker = self.ax_hist.axvline(self.cmin, color='w')
        self.high_marker = self.ax_hist.axvline(self.cmax, color='k')

        ax_hist.step(self.bin_centers, self.hist, color='r', lw=2, alpha=1.)
        self.ax_hist.set_xlim(self.cmin, self.cmax)
        self.ax_hist.set_xticks([])
        self.ax_hist.set_yticks([])

        slider_range = self.cmin, self.cmax
        self.slider_high = Slider(ax_high, slider_range, label='Max',
                                  value=self.cmax,
                                  on_release=self.update_image)
        self.slider_low = Slider(ax_low, slider_range, label='Min',
                                 value=self.cmin,
                                 on_release=self.update_image)
        self.slider_low.slidermax = self.slider_high
        self.slider_high.slidermin = self.slider_low

        # initialize histogram background
        imshow = self.ax_hist.imshow
        xmin, xmax, ymin, ymax = self.ax_hist.axis()
        self.black_bg = imshow(zeros((1, 2)), aspect='auto',
                               extent=(xmin, self.cmin, ymin, ymax))
        self.white_bg = imshow(ones((1, 2)), aspect='auto', vmin=0, vmax=1,
                               extent=(self.cmax, xmax, ymin, ymax))
        gradient = linspace(self.cmin, self.cmax, 256).reshape((1, 256))
        self.grad_bg = imshow(gradient, aspect='auto',
                              extent=(self.cmin, self.cmax, ymin, ymax))

        self.connect_event('key_press_event', self.on_key_press)
        self.connect_event('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.original_image = self.viewer.image.copy()
        self.update_image()
        print self.help

    @property
    def help(self):
        helpstr = ("ContrastSetter plugin\n"
                   "---------------------\n"
                   "+ and - keys or mouse scroll\n"
                   "to change the contrast\n")
        return helpstr

    @property
    def low(self):
        return self.slider_low.value

    @property
    def high(self):
        return self.slider_high.value

    def update_image(self, event=None):
        self.draw_background()
        self.viewer.climits = (self.low, self.high)
        self.viewer.redraw()
        self.redraw()

    def draw_background(self):
        xmin, xmax, ymin, ymax = self.ax_hist.axis()
        self.black_bg.set_extent((xmin, self.low, ymin, ymax))
        self.white_bg.set_extent((self.high, xmax, ymin, ymax))
        self.grad_bg.set_extent((self.low, self.high, ymin, ymax))
        self.low_marker.set_xdata([self.low, self.low])
        self.high_marker.set_xdata([self.high, self.high])

    def reset(self):
        self.slider_low.value = self.cmin
        self.slider_high.value = self.cmax
        self.update_image()

    def _expand_limits(self, event):
        if not event.inaxes: return
        span = self.high - self.low
        self.slider_low.value -= span / 20.
        self.slider_high.value += span / 20.
        self.update_image()

    def _restrict_limits(self, event):
        if not event.inaxes: return
        span = self.high - self.low
        self.slider_low.value += span / 20.
        self.slider_high.value -= span / 20.
        self.update_image()

    def on_scroll(self, event):
        if not event.inaxes: return
        if event.button == 'up':
            self._expand_limits(event)
        elif event.button == 'down':
            self._restrict_limits(event)

    def on_key_press(self, event):
        if not event.inaxes: return
        elif event.key == '+':
            self._expand_limits(event)
        elif event.key == '-':
            self._restrict_limits(event)
        elif event.key == 'r':
            self.reset()

