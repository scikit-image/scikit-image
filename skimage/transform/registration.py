import numpy as np
from scipy.ndimage import shift as image_shift

from .pyramids import pyramid_reduce

__all__ = ['register']


def _error(img1, img2):
    return np.square(np.subtract(img1, img2)).sum()/(255*len(img1)*len(img1[0]))


def register(img1, img2, exp=1.2):
    shift = [0, 0]
    loop = min(len(img1), len(img1[0]))

    while loop > 1:
        current = pyramid_reduce(img1, loop)

        val = 1
        loc = (0, 0)
        for i in range(-1, 2):
            for j in range(-1, 2):
                err = _error(pyramid_reduce(image_shift(
                    img2, (shift[0]+i*loop, shift[1]+j*loop)), loop), current)
                if val > err:
                    val = err
                    loc = (i*loop, j*loop)

        shift[0] += loc[0]
        shift[1] += loc[1]

        loop = loop//exp

    val = 1
    loc = (0, 0)
    for i in range(-1, 2):
        for j in range(-1, 2):
            err = _error(image_shift(img2, (shift[0]+i, shift[1]+j)), img1)
            if val > err:
                val = err
                loc = (i, j)

    shift[0] += loc[0]
    shift[1] += loc[1]

    return image_shift(img2, shift), shift
