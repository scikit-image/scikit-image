import numpy as np

# guard against import of a non-existant metrics module in older skimage
try:
    from skimage import metrics
except ImportError:
    pass


class SetMetricsSuite(object):
    shape = (6, 6)
    coords_a = np.zeros(shape, dtype=np.bool)
    coords_b = np.zeros(shape, dtype=np.bool)

    def setup(self):
        try:
            from skimage.metrics import hausdorff_distance
        except ImportError:
            raise NotImplementedError("hausdorff_distance unavailable")
        points_a = (1, 0)
        points_b = (5, 2)
        self.coords_a[points_a] = True
        self.coords_b[points_b] = True

    def time_hausdorff(self):
        metrics.hausdorff_distance(self.coords_a, self.coords_b)
