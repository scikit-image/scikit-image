import numpy as np

from skimage.color import hsv2rgb, rgb2hsv


class ColorSuite:
    """Benchmark for color space conversions in `skimage.color`."""

    param_names = ["shape"]
    params = [[(256, 256), (1024, 1024), (2048, 2048)]]

    def setup(self, shape):
        rng = np.random.default_rng(0)
        self.rgb = rng.random((*shape, 3))
        self.hsv = rgb2hsv(self.rgb)

    def time_rgb2hsv(self, shape):
        rgb2hsv(self.rgb)

    def time_hsv2rgb(self, shape):
        hsv2rgb(self.hsv)

    def peakmem_rgb2hsv(self, shape):
        rgb2hsv(self.rgb)

    def peakmem_hsv2rgb(self, shape):
        hsv2rgb(self.hsv)
