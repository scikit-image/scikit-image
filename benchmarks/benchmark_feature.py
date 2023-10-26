# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np
from skimage import color, data, feature, util


class FeatureSuite:
    """Benchmark for feature routines in scikit-image."""

    def setup(self):
        # Use a real-world image for more realistic features, but tile it to
        # get a larger size for the benchmark.
        self.image = np.tile(color.rgb2gray(data.astronaut()), (4, 4))
        self.image_ubyte = util.img_as_ubyte(self.image)
        self.keypoints = feature.corner_peaks(
            self.image, min_distance=5, threshold_rel=0.1
        )

    def time_canny(self):
        feature.canny(self.image)

    def time_glcm(self):
        pi = np.pi
        feature.greycomatrix(
            self.image_ubyte, distances=[1, 2], angles=[0, pi / 4, pi / 2, 3 * pi / 4]
        )

    def time_brief(self):
        extractor = feature.BRIEF()
        extractor.extract(self.image, self.keypoints)

    def time_hessian_matrix_det(self):
        feature.hessian_matrix_det(self.image, 4)
