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


.. [1] Simoncelli, E.P. & Freeman, W.T.
       (1995). The Steerable Pyramid: A Flexible Architecture for Multi-Scale
       Derivative Computation. In Proc. 2nd IEEE International Conf. on Image
       Proc., vol.III pp. 444-447, Oct 1995. 
       http://www.cns.nyu.edu/~eero/steerpyr/,
       http://www.cns.nyu.edu/pub/eero/simoncelli95b.pdf,
       DOI:10.1109/ICIP.1995.537667
"""

from __future__ import division, print_function
import numpy as np
from matplotlib import pyplot as plt
from skimage.transform import steerable
from skimage.exposure import rescale_intensity
from skimage.morphology import disk


def visualize(coeff):
    '''
    Generate a single image, which visualize
    Steerable subband decomposition 'coeff'
    Subbands of the same height are on the same row
    '''

    rows, cols = coeff[1][0].shape
    nr_orients = len(coeff[1])
    out = np.ones((rows * 2 - coeff[-1][0].shape[0] + 3,
                   nr_orients * cols + 3), dtype=np.double)

    r = 0
    for i in range(1, len(coeff[:-1])):
        m, n = coeff[i][0].shape

        c = 0
        for j in range(len(coeff[1])):
            subband = coeff[i][j].real
            subband = rescale_intensity(subband)

            subband[-1, :] = 1
            subband[:, -1] = 1

            out[r: r + m, c: c + n] = subband
            c += n
        r += m

    m, n = coeff[-1][0].shape
    out[r: r + m, 0:n] = rescale_intensity(coeff[-1][0])

    return out


# create an image of a disk
# x = np.arange(-128, 128)
# xx, yy = np.meshgrid(x, x, sparse=True)
# r = np.sqrt(xx**2 + yy**2)
# image = r < 64
image = disk(64)
print(image.shape)

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
