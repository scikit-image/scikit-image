"""
===============================================================================
Harris Corner detector
===============================================================================

The Harris corner filter detects interest points using edge detection in many
direction.
"""
from matplotlib import pyplot as plt
from matplotlib import cm

from skimage import data
from skimage.filter import harris


def plot_harris_points(image, filtered_coords):
    """ plots corners found in image"""

    plt.subplot(111)
    plt.imshow(image, cmap=cm.gray)
    plt.plot([p[1] for p in filtered_coords],
                [p[0] for p in filtered_coords],
                'b.')
    plt.axis('off')
    plt.show()


im = data.lena().astype(float)
filtered_coords = harris.harris_corner_detector(im, 6)
plot_harris_points(im, filtered_coords)
