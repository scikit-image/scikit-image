import numpy as np
import heapq
from skimage.morphology import dilation, disk


__all__ = ['initialise']


KNOWN = 0
BAND = 1
INSIDE = 2


def initialise(_mask):
    """Initialisation for Image Inpainting technique based on Fast Marching
    Method as outined in [1]_. Each pixel has 2 new values assigned to it
    stored in `flag` and `u` arrays.

    `flag` Initialisation:
    All pixels are classified into 1 of the following flags:
    # 0 = KNOWN - intensity and u values are known.
    # 1 = BAND - u value undergoes an update.
    # 2 = INSIDE - intensity and u values unkown

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
    _mask : array
        `True` values are to be inpainted.

    Returns
    ------
    flag : array of int
        Array marking pixels as known, along the boundary to be solved, or
        inside the unknown region: 0 = KNOWN, 1 = BAND, 2 = INSIDE
    u : (array of float
        The distance/time map from the boundary to each pixel.
    heap : list of tuples
        BAND points with heap element as mentioned above

    References
    ----------
    .. [1] Telea, A., "An Image Inpainting Technique based on the Fast Marching
            Method", Journal of Graphic Tools (2004).
            http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    """

    mask = _mask.astype(np.uint8)
    outside = dilation(mask, disk(1))
    band = np.logical_xor(mask, outside).astype(np.uint8)

    flag = (2 * outside) - band

    u = np.where(flag == INSIDE, 1.0e6, 0)

    heap = []
    indices = np.transpose(np.where(flag == BAND))
    for z in indices:
        heapq.heappush(heap, (u[tuple(z)], tuple(z)))

    return flag, u, heap
