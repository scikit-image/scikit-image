from skimage.filter import canny
from skimage.transform import probabilistic_hough

import matplotlib.pyplot as plt
import matplotlib.collections as mcoll

from .base import Plugin
from ..widgets.slider import Slider
from .. import utils


__all__ = ['HoughExplorer']



class HoughExplorer(Plugin):
    """Plugin to find line segments in an image using Hough Transform.

    Parameters
    ----------
    image_viewer : ImageViewer instance.
        Window containing image used in measurement.
    """


    def __init__(self, image_viewer,
                 sigma_range=(0, 5), low_range=(0, 1), high_range=(0, 1),
                 threshold_range=(0, 255), length_range=(1, 250),
                 gap_range=(0, 50), color='r'):

        with utils.toolbar_off():
            figure, axes = plt.subplots(nrows=6, figsize=(6, 2))
        Plugin.__init__(self, image_viewer, figure=figure)

        ax_sigma, ax_low, ax_high, ax_thresh, ax_length, ax_gap = axes
        # Adjust spacing on left to fit slider label
        figure.subplots_adjust(left=0.2)

        self.color = color

        # Canny parameters
        self.slider_sigma = Slider(ax_sigma, sigma_range, label='sigma',
                                   value=1, on_release=self.update_image)
        self.slider_low = Slider(ax_low, low_range, label='low',
                                 value=0.5, on_release=self.update_image)
        self.slider_high = Slider(ax_high, high_range, label='high',
                                  value=0.7, on_release=self.update_image)
        self.slider_low.slidermax = self.slider_high
        self.slider_high.slidermin = self.slider_low

        # Hough parameters
        self.slider_line_threshold = Slider(ax_thresh, threshold_range,
                                            label='line threshold', value=1,
                                            on_release=self.update_lines)
        self.slider_line_length = Slider(ax_length, length_range,
                                         label='line length', value=20,
                                         on_release=self.update_lines)
        self.slider_line_gap = Slider(ax_gap, gap_range,
                                      label='line gap', value=0,
                                      on_release=self.update_lines)

        self.original_image = self.viewer.image.copy()

        self._hough_lines = []

        self.canny_image = None
        self.update_image()

        print self.help


    @property
    def help(self):
        msg = ("HoughExplorer Plugin\n"
               "-------------------\n"
               "Adjust parameters to Canny edge detector and Hough transform:\n"
               "    sigma: smoothing factor (std of Gaussian)\n"
               "    low: minimum intensity for an edge value\n"
               "    high: threshold for *starting points* of edge detection\n"
               "    threshold: threshold probabilistic Hough transform\n"
               "    line_length: minimum length of line\n"
               "    line_gap: maximum gap between continuous lines\n")
        return msg

    @property
    def sigma(self):
        return self.slider_sigma.value

    @property
    def low(self):
        return self.slider_low.value

    @property
    def high(self):
        return self.slider_high.value

    @property
    def line_threshold(self):
        return self.slider_line_threshold.value

    @property
    def line_length(self):
        return self.slider_line_length.value

    @property
    def line_gap(self):
        return self.slider_line_gap.value

    def update_image(self, event=None):

        self.canny_image = canny(self.original_image,
                                 sigma=self.sigma,
                                 low_threshold=self.low,
                                 high_threshold=self.high)
        self.update_lines()

    def update_lines(self, event=None):

        if self._hough_lines:
            self.viewer.ax.collections.remove(self._hough_lines)

        lines = probabilistic_hough(self.canny_image,
                                    threshold=self.line_threshold,
                                    line_length=self.line_length,
                                    line_gap=self.line_gap)

        self._hough_lines = mcoll.LineCollection(lines, colors=self.color)
        self.viewer.ax.add_collection(self._hough_lines)
        self.viewer.redraw()

