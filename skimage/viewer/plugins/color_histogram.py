import numpy as np
import matplotlib.pyplot as plt

from skimage import color
from skimage import exposure
from .plotplugin import PlotPlugin
from ..canvastools import RectangleTool


class ColorHistogram(PlotPlugin):
    name = 'Color Histogram'

    def __init__(self, max_pct=0.99, **kwargs):
        super(ColorHistogram, self).__init__(height=400, **kwargs)
        self.max_pct = max_pct

        print(self.help())

    def attach(self, image_viewer):
        super(ColorHistogram, self).attach(image_viewer)

        self.rect_tool = RectangleTool(self.ax, on_release=self.ab_selected)
        self.lab_image = color.rgb2lab(image_viewer.image)

        # Calculate color histogram in the Lab colorspace:
        L, a, b = self.lab_image.T
        left, right = -100, 100
        ab_extents = [left, right, right, left]
        bins = np.arange(left, right)
        hist, x_edges, y_edges = np.histogram2d(a.flatten(), b.flatten(), bins,
                                                normed=True)

        # Clip bin heights that dominate a-b histogram
        max_val = pct_total_area(hist, percentile=self.max_pct)
        hist = exposure.rescale_intensity(hist, in_range=(0, max_val))
        self.ax.imshow(hist, extent=ab_extents, cmap=plt.cm.gray)

        self.ax.set_title('Color Histogram')
        self.ax.set_xlabel('b')
        self.ax.set_ylabel('a')

    def help(self):
        helpstr = ("Color Histogram tool:",
                   "Select region of a-b colorspace to highlight on image.")
        return '\n'.join(helpstr)

    def ab_selected(self, extents):
        x0, x1, y0, y1 = extents

        lab_masked = self.lab_image.copy()
        L, a, b = lab_masked.T

        mask = ((a > y0) & (a < y1)) & ((b > x0) & (b < x1))
        lab_masked[..., 1:][~mask.T] = 0

        self.image_viewer.image = color.lab2rgb(lab_masked)


def pct_total_area(image, percentile=0.80):
    """Return threshold value based on percentage of total area.

    The specified percent of pixels less than the given intensity threshold.
    """
    idx = int((image.size - 1) * percentile)
    sorted_pixels = np.sort(image.flat)
    return sorted_pixels[idx]



