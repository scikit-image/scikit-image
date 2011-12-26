"""
======================
Histogram Equalization
======================

This examples takes an image with low contrast and enhances its contrast using
histogram equalization. Histogram equalization enhances contrast by "spreading
out the most frequent intensity values" in an image [1]_. The equalized image
has a roughly linear cumulative distribution function, as shown in this example.

.. [1] http://en.wikipedia.org/wiki/Histogram_equalization

"""
import matplotlib.pyplot as plt

from skimage import data
from skimage.util.dtype import dtype_range
from skimage import exposure


def plot_hist(img, bins=256):
    """Plot histogram and cumulative histogram for image"""
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    plt.hist(img.ravel(), bins=bins)
    ax_cdf = plt.twinx()
    ax_cdf.plot(bins, img_cdf, 'r')
    xmin, xmax = dtype_range[img.dtype.type]
    plt.xlim(xmin, xmax)

    plt.ylabel('# pixels')
    plt.xlabel('pixel intensiy')
    ax_cdf.set_ylabel('fraction of total intensity')


img_orig = data.camera()
# squeeze image intensities to lower image contrast
img = img_orig / 5 + 100
img_eq = exposure.equalize(img)

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

