"""
===============================================================================
Harris Corner detector
===============================================================================

The Harris corner filter detects interest points using edge detection in
multiple direction.
"""

from matplotlib import pyplot as plt

from skimage import data, img_as_float
from skimage.filter import harris


def plot_harris_points(image, filtered_coords):
    """ plots corners found in image"""

    plt.plot()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],
             [p[0] for p in filtered_coords],
             'b.')
    plt.axis('off')
    plt.show()


im = img_as_float(data.lena())
filtered_coords = harris(im, 6)
plot_harris_points(im, filtered_coords)
