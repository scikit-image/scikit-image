__all__ = ['init_flag', 'init_u', 'generate_heap', 'display_heap']

import numpy as np
import heapq
from skimage.morphology import erosion, disk


BAND = 1


def init_flag(_mask):
    """Initialization:
    All pixels are classified into 1 of the following flags:
    # KNOWN - denoted by an integer value of 0
    # BAND - denoted by an integer value of 1
    # INSIDE - denoted by an integer value of 2

    Depending on the flag value, that is, if flag is BAND or KNOWN
    u is set to 0 and for flag equal to INSIDE, u is set to 1.0e6,
    arbitrarily large value.

    Parameters
    ----------
    mask : ndarray of bool
        This array is cast to uint8 and normalized to 1 before processing.

    Returns
    -------
    flag : ndarray of uint8
        It cosists of either 0, 1 or 2 according to conditions above.
    """

    mask = _mask.astype(np.uint8)
    inside = erosion(mask, disk(1))
    band = np.logical_xor(mask, inside).astype(np.uint8)

    flag = band + (2 * inside)

    return flag


def init_u(flag):
    return np.where(flag == BAND, 0, 1.0e6)


def generate_heap(flag, u):
    """Initialization:
    All pixels are classified into 1 of the following flags:
    # BAND - denoted by an integer value of 0
    # KNOWN - denoted by an integer value of 1
    # INSIDE - denoted by an integer value of 2

    Depending on the flag value, that is, if `flag` is BAND or KNOWN
    `u` or speed is set to 0 and for flag equal to INSIDE, `u` is set to 1.0e6,
    arbitrarily large value.

    Parameters
    ----------
    mask : ndarray
        Binary input image. This array is cast to uint8 and normalized to 1
        before processing.

    Returns
    -------
    heap : list of HeapData objects
        It consists of the `u` or speed values and the index.
    """

    heap = []

    indices = np.transpose(np.where(flag == BAND))
    for z in indices:
        heapq.heappush(heap, (u[tuple(z)], z))

    return heap


def display_heap(heap):
    for i in heap:
        print i[0], i[1]
