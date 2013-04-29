# TODO : Doc, Tests, PEP8 check

import numpy as np

def downsample(image, factors, method='sum'):

    is0 = image.shape[0]
    is1 = image.shape[1]
    f0 = factors[0]
    f1 = factors[1]

    if (f0 - int(f0) != 0) or (f1 - int(f1) != 0):
        print "Use resample() for non-integer downsampling"
        return
    cropped = image[:is0 - (is0 % f0), :is1 - (is1 % f1)]
    out = np.zeros((cropped.shape[0] / f0, cropped.shape[1] / f1))

    for i in range(cropped.shape[0]):
        for j in range(cropped.shape[1]):
            out[int(i / f0)][int(j / f1)] += cropped[i][j]
    if method == 'sum':
        return out
    else:
        return out / float(f0 * f1)


def upsample(image, factors, method='divide'):

    is0 = image.shape[0]
    is1 = image.shape[1]
    f0 = factors[0]
    f1 = factors[1]

    if (f0 - int(f0) != 0) or (f1 - int(f1) != 0):
        print "Use resample() for non-integer upsampling"
        return
    out = np.zeros((f0 * image.shape[0], f1 * image.shape[1]))


    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i][j] = (image[i / f0][j / f1])
    if method == 'divide':
        return out / float(f0 * f1)
    else:
        return out
