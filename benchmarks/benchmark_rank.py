import numpy as np
from skimage.filters import rank
from skimage.filters.rank import __all__ as all_rank_filters
from skimage.filters.rank import __3Dfilters as all_3d_rank_filters
from skimage.morphology import disk, ball


class RankSuite:
    param_names = ["filter_func", "shape"]
    params = [sorted(all_rank_filters), [(32, 32), (256, 256)]]

    def setup(self, filter_func, shape):
        self.image = np.random.randint(0, 255, size=shape, dtype=np.uint8)
        self.footprint = disk(1)

    def time_filter(self, filter_func, shape):
        getattr(rank, filter_func)(self.image, self.footprint)


class Rank3DSuite:
    param_names = ["filter3d", "shape3d"]
    params = [sorted(all_3d_rank_filters), [(32, 32, 32), (128, 128, 128)]]

    def setup(self, filter3d, shape3d):
        self.volume = np.random.randint(0, 255, size=shape3d, dtype=np.uint8)
        self.footprint_3d = ball(1)

    def time_3d_filters(self, filter3d, shape3d):
        getattr(rank, filter3d)(self.volume, self.footprint_3d)
