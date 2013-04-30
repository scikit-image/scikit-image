# TODO : Doc, Tests, PEP8 check

import numpy as np
from skimage.util.shape import view_as_blocks

def downsample(image, factors, method='sum'):

    # works only if image.shape is perfectly divisible by factors
    out = view_as_blocks(image, factors)
    block_shape = out.shape

    if method == 'sum':
        for i in range(len(block_shape)/2):
            out = out.sum(-1)
    else:
        for i in range(len(block_shape)/2):
            out = out.mean(-1)
    return out

def upsample(image, factors, method='divide'):

    f = factors

    if (f[0] - int(f[0]) != 0) or (f[1] - int(f[1]) != 0):
        raise ValueError('Use resample() for non-integer upsampling')
    out = np.zeros((f[0] * image.shape[0], f[1] * image.shape[1]))

    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i][j] = (image[i / f[0]][j / f[1]])
    if method == 'divide':
        return out / float(f[0] * f[1])
    else:
        return out
