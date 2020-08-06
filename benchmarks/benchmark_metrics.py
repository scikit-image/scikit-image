import numpy as np

try:
    from skimage import metrics
    have_metrics_module = True
except ImportError:
    have_metrics_module = False


class SetMetricsSuite(object):
    shape = (6, 6)
    coords_a = np.zeros(shape, dtype=np.bool)
    coords_b = np.zeros(shape, dtype=np.bool)

    def setup(self):
        points_a = (1, 0)
        points_b = (5, 2)
        self.coords_a[points_a] = True
        self.coords_b[points_b] = True

    def time_hausdorff(self):
        if not have_metrics_module:
            return NotImplementedError("metrics module unavailable")
        metrics.hausdorff_distance(self.coords_a, self.coords_b)
