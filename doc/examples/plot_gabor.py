"""
=============================================
Gabor filter banks for texture classification
=============================================

In this example, we will see how to classify textures based on Gabor filter
banks. Frequency and orientation representations of the Gabor filter are similar
to those of the human visual system.

The images are filtered using the real parts of various different Gabor filter
kernels. The mean and variance of the filtered images are then used as features
for classification, which is based on the least squared error for simplicity.

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as nd
from skimage import data
from skimage.util import img_as_float
from skimage.filter import gabor_kernel


matplotlib.rcParams['font.size'] = 9


def compute_feats(image, kernels):
    feats = np.zeros((len(kernels), 2), dtype=np.double)
    for k, kernel in enumerate(kernels):
        filtered = nd.convolve(image, kernel, mode='wrap')
        feats[k, 0] = filtered.mean()
        feats[k, 1] = filtered.var()
    return feats


def match(feats, ref_feats):
    min_error = np.inf
    min_i = None
    for i in range(ref_feats.shape[0]):
        error = np.sum((feats - ref_feats[i, :])**2)
        if error < min_error:
            min_error = error
            min_i = i
    return min_i


# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(sigma, sigma, frequency, theta))
            kernels.append(kernel)


brick = img_as_float(data.load('brick.png'))
grass = img_as_float(data.load('grass.png'))
wall = img_as_float(data.load('rough-wall.png'))
image_names = ('brick', 'grass', 'wall')

# prepare refernce features
ref_feats = np.zeros((3, len(kernels), 2), dtype=np.double)
ref_feats[0, :, :] = compute_feats(brick, kernels)
ref_feats[1, :, :] = compute_feats(grass, kernels)
ref_feats[2, :, :] = compute_feats(wall, kernels)


print 'Rotated images matched against references using Gabor filter banks:'

print 'original: brick, rotated: 30deg, match result:',
feats = compute_feats(nd.rotate(brick, angle=190, reshape=False), kernels)
print image_names[match(feats, ref_feats)]

print 'original: brick, rotated: 70deg, match result:',
feats = compute_feats(nd.rotate(brick, angle=70, reshape=False), kernels)
print image_names[match(feats, ref_feats)]

print 'original: grass, rotated: 145deg, match result:',
feats = compute_feats(nd.rotate(grass, angle=145, reshape=False), kernels)
print image_names[match(feats, ref_feats)]


# plot a selection of the filter bank kernels

kernels = []
kernel_params = []
for theta in (0, 1, 3):
    theta = theta / 4. * np.pi
    for frequency in (0.05, 0.1, 0.25):
        kernel = np.real(gabor_kernel(10, 10, frequency, theta))
        kernels.append(kernel)
        params = 'theta=%d, frequency=%.2f' % (theta * 180 / np.pi, frequency)
        kernel_params.append(params)


fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3,
                                                       figsize=(9, 6))
plt.gray()

fig.text(.5, .95, 'Gabor filter bank kernels',
         horizontalalignment='center', fontsize=15)

for i, ax in enumerate((ax1, ax2, ax3, ax4, ax5, ax6)):
    ax.imshow(kernels[i], interpolation='nearest')
    ax.axis('off')
    ax.set_title(kernel_params[i])

plt.show()
