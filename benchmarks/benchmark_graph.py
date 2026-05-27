# See "Writing benchmarks" in the asv docs for more information.
# https://asv.readthedocs.io/en/latest/writing_benchmarks.html
import numpy as np

from scipy import ndimage as ndi
from skimage import color, data, filters, graph, morphology


class GraphSuite:
    """Benchmark for pixel graph routines in scikit-image."""

    def setup(self):
        retina = color.rgb2gray(data.retina())
        t0, _ = filters.threshold_multiotsu(retina, classes=3)
        mask = retina > t0
        vessels = filters.sato(retina, sigmas=range(1, 10)) * mask
        thresholded = filters.apply_hysteresis_threshold(vessels, 0.01, 0.03)
        labeled = ndi.label(thresholded)[0]
        largest_nonzero_label = np.argmax(np.bincount(labeled[labeled > 0]))
        binary = labeled == largest_nonzero_label
        self.skeleton = morphology.skeletonize(binary)

        labeled2 = ndi.label(thresholded[::2, ::2])[0]
        largest_nonzero_label2 = np.argmax(np.bincount(labeled2[labeled2 > 0]))
        binary2 = labeled2 == largest_nonzero_label2
        small_skeleton = morphology.skeletonize(binary2)
        self.g, self.n = graph.pixel_graph(small_skeleton, connectivity=2)

    def time_build_pixel_graph(self):
        graph.pixel_graph(self.skeleton, connectivity=2)

    def time_central_pixel(self):
        graph.central_pixel(self.g, self.n)
