"""
======================
Histogram Equalization
======================

This examples takes an image with low contrast and enhances its contrast using
histogram equalization. Histogram equalization enhances contrast by "spreading
out the most frequent intensity values" in an image [1]. The equalized image
has a roughly linear cumulative distribution function, as shown in this example.

.. [1] http://en.wikipedia.org/wiki/Histogram_equalization

"""
import matplotlib.pyplot as plt

from skimage import data
from skimage.util.dtype import dtype_range
from skimage import exposure


def plot_hist(img, bins=256, ax=None):
    """Plot histogram and cumulative histogram for image"""
    ax = ax if ax is not None else plt.gca()
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    ax.hist(img.ravel(), bins=bins)
    ax_right = ax.twinx()
    ax_right.plot(bins, img_cdf, 'r')
    xmin, xmax = dtype_range[img.dtype.type]
    ax.set_xlim(xmin, xmax)

    ax.set_ylabel('# pixels')
    ax.set_xlabel('pixel intensiy')
    ax_right.set_ylabel('fraction of total intensity')


img_orig = data.camera()
# squeeze image intensities to lower image contrast
img = img_orig / 5 + 100
img_eq = exposure.equalize_hist(img)

plt.subplot(2, 2, 1)
plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.axis('off')
plt.subplot(2, 2, 2)
plot_hist(img)

plt.subplot(2, 2, 3)
plt.imshow(img_eq, cmap=plt.cm.gray, vmin=0, vmax=1)
plt.axis('off')
plt.subplot(2, 2, 4)
plot_hist(img_eq)

plt.subplots_adjust(left=0.05, hspace=0.25, wspace=0.3, top=0.95, bottom=0.1)
plt.show()

