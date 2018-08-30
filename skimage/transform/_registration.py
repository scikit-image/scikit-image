import numpy as np
from scipy.ndimage import shift as image_shift

from ..data import camera
from ._warps import downscale_local_mean
from .pyramids import pyramid_reduce

def _error(img1,img2):
    return np.square(np.subtract(img1,img2)).sum()/(255*len(img1)*len(img1[0]))

def register(img1, img2):
    shift = [0,0]
    loop = min(len(img1), len(img2[0]))

    temp2 = img2
    while loop > 1:
        current = pyramid_reduce(img1,loop)
        temp = pyramid_reduce(temp2,loop)

        val = 1
        loc = (0,0)
        for i in range(-1,2):
            for j in range(-1,2):
                err = _error(image_shift(temp,(i,j)),current)
                if val > err:
                    val = err
                    loc = (i*loop,j*loop)

        shift[0] += loc[0]
        shift[1] += loc[1]
        temp2 = image_shift(img2,shift)

        loop = loop//2

    return temp2, shift
    