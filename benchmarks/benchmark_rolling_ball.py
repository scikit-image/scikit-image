# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html

import numpy as np

from skimage import morphology
from skimage import data


class RollingBallSuite:
    """Benchmark for Rolling Ball algorithm in scikit-image."""
    params = [25, 50, 75, 100, 150, 200]

    def setup(self):
        self.black_bg = data.coins()

    def time_execution(self, radius):
        rolling_ball(self.black_bg, radius=radius)

    def peakmem_execution(self, radius):
        rolling_ball(self.black_bg, radius=radius)
