import numpy as np
from scipy.ndimage.filters import maximum_filter

from fast_cy import _corner_fast 


def corner_fast(image, n=12, threshold=0.15):

    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Only 2-D gray-scale images supported.")

    image = np.ascontiguousarray(image, dtype=np.double)
    corner = _corner_fast(image, n, threshold)

    corner_zero_mask = corner != 0
    c, d = np.nonzero(corner)

    maximas = (maximum_filter(corner, (3, 3)) == corner) & corner_zero_mask
    x, y = np.where(maximas == True)

    return np.squeeze(np.dstack((x, y)))
