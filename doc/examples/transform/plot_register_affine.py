"""
=====================================
Affine Registration
=====================================

In this example, we use image registration to find an affine transformation
which can be used to align our target and reference images.

The ``register_affine`` function uses a Gaussian pyramid and an
optimisation function to find the optimal transformation. This optimal
transformation, when applied to the target image, aligns it to the reference image [1]_.

.. [1] Juan Nunez-Iglesias, Stefan van der Walt, and Harriet Dashnow. Elegant
SciPy: The Art of Scientic Python. 1st. O'Reilly Media, Inc., 2017. isbn:
1491922877, 9781491922873.

"""

import numpy as np
from scipy import ndimage as ndi
from matplotlib import pyplot as plt

from skimage.data import camera
from skimage.transform import register_affine
from skimage import measure

intermediates_list = []


def save_intermediate_alignments(image, matrix):
    intermediates_list.append((image, matrix))


r = 0.12
c, s = np.cos(r), np.sin(r)
matrix_transform = np.array([[c, -s, 0], [s, c, 50], [0, 0, 1]])

image = camera()
target = ndi.affine_transform(image, matrix_transform)
register_matrix = register_affine(image, target, iter_callback=save_intermediate_alignments)
_, ax = plt.subplots(1, 6)

ax[0].set_title('reference')
ax[0].imshow(image, cmap='gray')
y, x = image.shape
ax[0].set_xticks(np.arange(x/5, x, x/5), minor=True)
ax[0].set_yticks(np.arange(y/5, y, y/5), minor=True)
ax[0].grid(which='minor', color='w', linestyle='-', linewidth=1)

err = measure.compare_mse(image, target)
ax[1].set_title('target, mse %d' % int(err))
ax[1].imshow(target, cmap='gray')
y, x = target.shape
ax[1].set_xticks(np.arange(x/5, x, x/5), minor=True)
ax[1].set_yticks(np.arange(y/5, y, y/5), minor=True)
ax[1].grid(which='minor', color='w', linestyle='-', linewidth=1)

for a in ax:
    a.set_xticklabels([])
    a.set_yticklabels([])

for i, iter_num in enumerate([1, 2, 4]):
    err = measure.compare_mse(
        image, ndi.affine_transform(target, intermediates_list[iter_num][1]))
    ax[i+2].set_title('iter %d, mse %d' % (iter_num, int(err)))
    ax[i+2].imshow(ndi.affine_transform(intermediates_list[iter_num][0], intermediates_list[iter_num][1]), cmap='gray',
                   interpolation='gaussian', resample=True)

    y, x = intermediates_list[iter_num][0].shape

    ax[i+2].set_xticks(np.arange(x/5, x, x/5), minor=True)
    ax[i+2].set_yticks(np.arange(y/5, y, y/5), minor=True)
    ax[i+2].grid(which='minor', color='w', linestyle='-', linewidth=1)

err = measure.compare_mse(image, ndi.affine_transform(target, register_matrix))
ax[5].set_title('final correction, mse %d' % int(err))
ax[5].imshow(ndi.affine_transform(target, register_matrix), cmap='gray',
             interpolation='gaussian', resample=True)
y, x = target.shape

ax[5].set_xticks(np.arange(x/5, x, x/5), minor=True)
ax[5].set_yticks(np.arange(y/5, y, y/5), minor=True)
ax[5].grid(which='minor', color='w', linestyle='-', linewidth=1)
plt.show()
