# TODO : Doc, Tests, PEP8 check

import numpy as np

def downsample(image, factors, method='sum'):

    is = image.shape
    f = factors

    if (f[0] - int(f[0]) != 0) or (f[1] - int(f[1]) != 0):
        raise ValueError('Use resample() for non-integer downsampling')
    cropped = image[:is[0] - (is[0] % f[0]), :is[1] - (is[1] % f[1])]
    out = np.zeros((cropped.shape[0] / f[0], cropped.shape[1] / f[1]))

    for i in range(cropped.shape[0]):
        for j in range(cropped.shape[1]):
            out[int(i / f[0])][int(j / f[1])] += cropped[i][j]
    if method == 'sum':
        return out
    else:
        return out / float(f[0] * f[1])


def upsample(image, factors, method='divide'):

    is = image.shape
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
