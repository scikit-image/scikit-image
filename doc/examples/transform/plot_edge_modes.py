"""
=========================
Interpolation: Edge Modes
=========================

This example illustrates the different edge modes available during
interpolation in routines such as :py:func:`skimage.transform.rescale`
and :py:func:`skimage.transform.resize`.
"""
import numpy as np
import matplotlib.pyplot as plt

from skimage.util import pad


img = np.zeros((16, 16))
img[:8, :8] += 1
img[:4, :4] += 1
img[:2, :2] += 1
img[:1, :1] += 2
img[8, 8] = 4

modes = ['constant', 'edge', 'wrap', 'reflect', 'symmetric']
fig, axes = plt.subplots(2, 3)
ax = axes.flatten()

for n, mode in enumerate(modes):
    img_padded = pad(img, pad_width=img.shape[0], mode=mode)
    ax[n].imshow(img_padded, cmap=plt.cm.gray, interpolation='nearest')
    ax[n].plot([15.5, 15.5, 31.5, 31.5, 15.5],
               [15.5, 31.5, 31.5, 15.5, 15.5], 'y--', linewidth=0.5)
    ax[n].set_title(mode)

for a in ax:
    a.set_axis_off()
    a.set_aspect('equal')

plt.tight_layout()
plt.show()
