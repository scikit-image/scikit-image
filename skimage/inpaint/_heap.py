__all__ = ['initialise']

import numpy as np
import heapq
from skimage.morphology import dilation, disk


KNOWN = 0
BAND = 1
INSIDE = 2


def initialise(_mask, flag, u, heap):
    """Initialisation:
    Each pixel has 2 new values assigned to it stored in `flag` and `u` arrays.

    `flag` Initialisation:
    All pixels are classified into 1 of the following flags:
    # KNOWN - denoted by 0 - intensity and u values are known.
    # BAND - denoted by 1 - u value undergoes an update.
    # INSIDE - denoted by 2 - intensity and u values unkown

    `u` Initialisation:
    u <- 0 : `flag` equal to BAND or KNOWN
    u <- 1.0e6 (arbitrarily large value) : `flag` equal to INSIDE

    `heap` Initialisation:
    Contains all the pixels marked as BAND in `flag`. The heap element is
    a tuple with 2 elements, first being the `u` value corresponding to the
    tuple of index which is stored as the second element.
    Heap Element : (u[(i, j)], (i, j))

    Parameters
    ----------
    _mask : ndarray of bool
        This array is cast to uint8. Suppose the size is (m, n)

    flag, u : ndarray of zeros of size (m+2, n+2)
        They contain the results after the initialisation as above.

    heap : Empty list
        Contains the BAND points with heap element as mentioned above

    """

    mask = _mask.astype(np.uint8)
    outside = dilation(mask, disk(1))
    band = np.logical_xor(mask, outside).astype(np.uint8)

    flag = (2 * outside) - band

    indices = np.transpose(np.where(flag == BAND))
    for z in indices:
        heapq.heappush(heap, (u[tuple(z)], tuple(z)))
