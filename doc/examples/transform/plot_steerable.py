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


.. [1] http://http://www.cns.nyu.edu/~eero/steerpyr/

"""
from matplotlib import pyplot as plt
import numpy as np

from skimage.data import chelsea
from skimage.color import rgb2gray
from skimage.transform import steerable


def visualize(coeff, normalize=True):
    rows, cols = coeff[1][0].shape
    Norients = len(coeff[1])
    out = np.zeros((rows * 2 - coeff[-1].shape[0] + 1, Norients * cols))

    r = 0
    c = 0
    for i in range(1, len(coeff[:-1])):
        for j in range(len(coeff[1])):
            subband = coeff[i][j].real
            m, n = subband.shape

            if normalize:
                subband = 255 * subband / subband.max()

            subband[m - 1, :] = 255
            subband[:, n - 1] = 255

            out[r: r + m, c: c + n] = subband
            c += n
        r += coeff[i][0].shape[0]
        c = 0

    m, n = coeff[-1].shape
    out[r: r + m, c: c + n] = \
        255 * coeff[-1] / coeff[-1].max()

    out[0, :] = 255
    out[:, 0] = 255

    return out

# create an image of a disk
x = np.arange(-128, 128)
xx, yy = np.meshgrid(x, x, sparse=True)
r = np.sqrt(xx**2 + yy**2)
image = r < 64

coeff = steerable.build_steerable(image)
out = visualize(coeff)

fig, ax = plt.subplots()
ax.imshow(out, cmap=plt.cm.gray)
ax.set_title("Subbands from Steerable decomposition")
plt.tight_layout()
plt.show()
