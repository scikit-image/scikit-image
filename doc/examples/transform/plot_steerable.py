"""
=================
Steerable Pyramid
=================

The function `steerable.build_steerable` takes a gray scale image
as input, and returns a list of lists of subbands.

The first element of the list is a numpy array represents highpass.
The last element of the list is a numpy array represents lowpass.
Intermediate elements are lists of subband at the same radius level
but different orientations.


Steerable pyramid decomposition
.. [1] http://http://www.cns.nyu.edu/~eero/steerpyr/

"""
from matplotlib import pyplot as plt
import numpy as np

from skimage.data import chelsea
from skimage.color import rgb2gray
from skimage.transform import steerable
from skimage.io import imshow
from skimage import img_as_ubyte


def visualize(coeff, normalize=True):
    rows, cols = coeff[1][0].shape
    Norients = len(coeff[1])
    out = np.zeros((rows * 2 - coeff[-1].shape[0] + 1, Norients * cols))

    currentx = 0
    currenty = 0
    for i in range(1, len(coeff[:-1])):
        for j in range(len(coeff[1])):
            tmp = coeff[i][j].real
            m, n = tmp.shape

            if normalize:
                tmp = 255 * tmp / tmp.max()

            tmp[m - 1, :] = 255
            tmp[:, n - 1] = 255

            out[currentx: currentx + m, currenty: currenty + n] = tmp
            currenty += n
        currentx += coeff[i][0].shape[0]
        currenty = 0

    m, n = coeff[-1].shape
    out[currentx: currentx + m, currenty: currenty +
        n] = 255 * coeff[-1] / coeff[-1].max()

    out[0, :] = 255
    out[:, 0] = 255

    return out


image = chelsea()
image = rgb2gray(image)

coeff = steerable.build_steerable(image)
out = visualize(coeff)

fig, ax = plt.subplots()
ax.imshow(out, cmap='gray')
plt.show()
