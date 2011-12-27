"""
======================
Histogram Equalization
======================

This examples takes an image with low contrast and enhances its contrast using
histogram equalization. Histogram equalization enhances contrast by "spreading
out the most frequent intensity values" in an image [1]_. The equalized image
has a roughly linear cumulative distribution function, as shown in this example.

For comparison, this example also shows an image after its intensity levels are
uniformly stretched.

.. [1] http://en.wikipedia.org/wiki/Histogram_equalization

"""
import matplotlib.pyplot as plt
from matplotlib import ticker

from skimage import data
from skimage.util.dtype import dtype_range
from skimage import exposure


def plot_hist(img, bins=256):
    """Plot histogram and cumulative histogram for image"""
    img_cdf, bins = exposure.cumulative_distribution(img, bins)
    plt.hist(img.ravel(), bins=bins)
    plt.xlabel('Pixel intensiy')
    # Shorten y-tick labels using scientific notation
    y_formatter = ticker.ScalarFormatter(useOffset=True)
    y_formatter.set_powerlimits((0, 0)) # force use of scientific notation
    ax = plt.gca()
    ax.yaxis.set_major_formatter(y_formatter)

    ax_cdf = plt.twinx()
    ax_cdf.plot(bins, img_cdf, 'r')
    xmin, xmax = dtype_range[img.dtype.type]
    plt.xlim(xmin, xmax)


img_orig = data.camera()
# squeeze image intensities to lower image contrast
img = img_orig / 5 + 100
img_rescale = exposure.rescale_intensity(img)
img_eq = exposure.equalize(img)

plt.subplot(2, 3, 1)
plt.imshow(img, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.title('Low contrast image')
plt.axis('off')
plt.subplot(2, 3, 4)
plt.ylabel('Number of pixels')
plot_hist(img)

plt.subplot(2, 3, 2)
plt.imshow(img_rescale, cmap=plt.cm.gray, vmin=0, vmax=255)
plt.title('Rescale intensities')
plt.axis('off')
plt.subplot(2, 3, 5)
plot_hist(img_rescale)

plt.subplot(2, 3, 3)
plt.imshow(img_eq, cmap=plt.cm.gray, vmin=0, vmax=1)
plt.title('Histogram equalization')
plt.axis('off')
plt.subplot(2, 3, 6)
plot_hist(img_eq)
plt.ylabel('Fraction of total intensity')

# prevent overlap of y-axis labels
plt.subplots_adjust(wspace=0.4)
plt.show()

