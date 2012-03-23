from skimage.filter import canny
import matplotlib.pyplot as plt

from .base import Plugin
from ..widgets.slider import Slider
from .. import utils


__all__ = ['EdgeDetector']



class EdgeDetector(Plugin):
    """Plugin to detect edges in an image using a Sobel detector.

    Parameters
    ----------
    image_window : ImageViewer instance.
        Window containing image used in measurement.
    """


    def __init__(self, image_window, sigma_range=(0, 5), low_range=(0, 1),
                 high_range=(0, 1)):

        with utils.toolbar_off():
            figure, axes = plt.subplots(nrows=3, figsize=(6, 1))
        ax_sigma, ax_low, ax_high = axes
        Plugin.__init__(self, image_window, figure=figure)

        self.slider_sigma = Slider(ax_sigma, sigma_range, label='sigma',
                                   value=1, on_release=self.update_image)
        self.slider_low = Slider(ax_low, low_range, label='low',
                                 value=0.5, on_release=self.update_image)
        self.slider_high = Slider(ax_high, high_range, label='high',
                                  value=0.7, on_release=self.update_image)
        self.slider_low.slidermax = self.slider_high
        self.slider_high.slidermin = self.slider_low

        self.original_image = self.imgview.image.copy()
        self.update_image()

        print self.help


    @property
    def help(self):
        msg = ("Edgedetector Plugin\n"
               "-------------------\n"
               "Adjust parameters to Canny edge detector:\n"
               "    sigma: smoothing factor (std of Gaussian)\n"
               "    low: minimum intensity for an edge value\n"
               "    high: threshold for *starting points* of edge detection")
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

    def update_image(self, event=None):

        self.imgview.image = canny(self.original_image,
                                   sigma=self.sigma,
                                   low_threshold=self.low,
                                   high_threshold=self.high)
        self.imgview.redraw()

