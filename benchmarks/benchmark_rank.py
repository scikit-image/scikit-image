import numpy as np
from skimage import img_as_ubyte
from skimage.filters import rank
from skimage.filters.rank import __all__ as all_rank_filters
from skimage.filters.rank import __3Dfilters__ as all_3d_rank_filters
from skimage.morphology import grey, disk, ball


class RankSuite(object):

    param_names = ["filter", "shape", "filter3d", "shape3d"]
    params = [sorted(all_rank_filters), [(32, 32), (256, 256)],
              sorted(all_3d_rank_filters), [(32, 32, 32), (256, 256, 256)]]

    def setup(self, filter, shape, filter3d, shape3d):
        self.image = np.random.randint(0, 255, size=shape, dtype=np.uint8)
        self.volume = np.random.randint(0, 255, size=shape3d, dtype=np.uint8)
        self.selem = disk(1)
        self.selem_3d = ball(1)

    def time_filter(self, filter, shape, filter3d, shape3d):
        getattr(rank, filter)(self.image, self.selem)

    def time_3d_filters(self, filter, shape, filter3d, shape3d):
        getattr(rank, filter3d)(self.volume, self.selem_3d)
