import numpy as np
from skimage import img_as_ubyte
from skimage.filters import rank
from skimage.filters.rank import __all__ as all_rank_filters
from skimage.morphology import grey, disk


class RankSuite(object):

    param_names = ["filter", "shape"]
    params = [sorted(all_rank_filters), [(32, 32), (256, 256)]]

    def setup(self, filter, shape):
        self.image = np.random.randint(0, 255, size=shape, dtype=np.uint8)
        self.selem = disk(1)

    def time_filter(self, filter, shape):
        getattr(rank, filter)(self.image, self.selem)
