"""
=================
Steerable Pyramid
=================

The steerable pyramid is a multi-resolution image decomposition
with directional selectivity [1]_.

The function `steerable.build_steerable` takes a gray scale image
as input, and returns a list of lists of subbands.

The first element of the list is a numpy array represents highpass.
The last element of the list is a numpy array represents lowpass.
Intermediate elements are lists of subband at the same radius level
but different orientations.


.. [1] E. P. Simoncelli and W. T. Freeman
    "The Steerable Pyramid: A Flexible Architecture
    for Multi-Scale Derivative Computation."
    http://www.cns.nyu.edu/~eero/steerpyr/
"""
from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import steerable


def normalize(im):
    return (im - im.min()) / (im.max() - im.min())


def visualize(coeff):
    rows, cols = coeff[1][0].shape
    Norients = len(coeff[1])
    out = np.ones((rows * 2 - coeff[-1][0].shape[0] + 1,
                   Norients * cols + 1), dtype=np.double)

    r = 0
    for i in range(1, len(coeff[:-1])):
        m, n = coeff[i][0].shape

        c = 0
        for j in range(len(coeff[1])):
            subband = coeff[i][j].real
            subband = normalize(subband)

            subband[-1, :] = 1
            subband[:, -1] = 1

            out[r: r + m, c: c + n] = subband
            c += n
        r += m

    m, n = coeff[-1][0].shape
    out[r: r + m, 0:n] = normalize(coeff[-1][0])

    return out


# create an image of a disk
x = np.arange(-128, 128)
xx, yy = np.meshgrid(x, x, sparse=True)
r = np.sqrt(xx**2 + yy**2)
image = r < 64


# Steerable subband decomposition
coeff = steerable.build_steerable(image)

print("Shape of Steerable subbands")
for i in range(len(coeff)):
    c = coeff[i]
    print("Height %d: " % i, end='')
    for array in c:
        print(array.shape, end=' ')
    print("")

out = visualize(coeff)

fig, ax = plt.subplots()
ax.imshow(out, cmap=plt.cm.gray)
ax.set_title("Subbands from Steerable decomposition")
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.show()
