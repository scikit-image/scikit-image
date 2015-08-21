"""
=========================
Interpolation: Edge Modes
=========================

This example illustrates the different edge modes available during
interpolation in routines such as `skimage.transform.rescale` and
`skimage.transform.resize`.
"""
from skimage._shared.interpolation import extend_image
import skimage.data
import matplotlib.pyplot as plt
import numpy as np

img = np.zeros((16, 16))
img[:8, :8] += 1
img[:4, :4] += 1
img[:2, :2] += 1
img[:1, :1] += 2
img[8, 8] = 4

modes = ['constant', 'edge', 'wrap', 'reflect', 'symmetric']
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
for n, mode in enumerate(modes):
    img_extended = extend_image(img, pad=img.shape[0], mode=mode)
    axes[n].imshow(img_extended, cmap=plt.cm.gray, interpolation='nearest')
    axes[n].plot([15.5, 15.5], [15.5, 31.5], 'y--', linewidth=0.5)
    axes[n].plot([31.5, 31.5], [15.5, 31.5], 'y--', linewidth=0.5)
    axes[n].plot([15.5, 31.5], [15.5, 15.5], 'y--', linewidth=0.5)
    axes[n].plot([15.5, 31.5], [31.5, 31.5], 'y--', linewidth=0.5)
    axes[n].set_axis_off()
    axes[n].set_aspect('equal')
    axes[n].set_title(mode)

plt.tight_layout()

plt.show()
