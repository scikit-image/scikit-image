__all__ = ['generate_heap', 'display_heap']

import numpy as np
import heapq
from skimage.morphology import erosion, disk


BAND = 0


def generate_flags(mask):
    """Initialization:
    All pixels are classified into 1 of the following flags:
    # BAND - denoted by an integer value of 0
    # KNOWN - denoted by an integer value of 1
    # INSIDE - denoted by an integer value of 2

    Depending on the flag value, that is, if flag is BAND or KNOWN
    T is set to 0 and for flag equal to INSIDE, T is set to 1.0e6,
    arbitrarily large value.

    Parameters
    ----------
    mask : ndarray
        This array is cast to uint8 and normalized to 1 before processing.

    Returns
    -------
    flag : ndarray of uint8
        It cosists of either 0, 1 or 2 according to conditions above.
    """

    mask = np.where(mask, 1, 0).astype(np.uint8)
    inside = erosion(mask, disk(1))
    border = np.logical_xor(mask, inside).astype(np.uint8)

    flag = 1 - border + inside
    u = inside * 1.0e6

    return flag, u


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

# @total_ordering
# class HeapElem(object):
#     def __init__(self, t, index):
#         self.t, self.index = t, tuple(index)

#     def __le__(self, other):
#         if not isinstance(other, type(self)):
#             return NotImplemented
#         return self.t <= other.t
