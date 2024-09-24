# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np
from skimage import util


class NoiseSuite:
    """Benchmark for noise routines in scikit-image."""

    params = ([0.0, 0.50, 1.0], [0.0, 0.50, 1.0])

    def setup(self, *_):
        self.image = np.zeros((5000, 5000))

    def peakmem_salt_and_pepper(self, amount, salt_vs_pepper):
        self._make_salt_and_pepper_noise(amount, salt_vs_pepper)

    def time_salt_and_pepper(self, amount, salt_vs_pepper):
        self._make_salt_and_pepper_noise(amount, salt_vs_pepper)

    def _make_salt_and_pepper_noise(self, amount, salt_vs_pepper):
        util.random_noise(
            self.image,
            mode="s&p",
            amount=amount,
            salt_vs_pepper=salt_vs_pepper,
        )
