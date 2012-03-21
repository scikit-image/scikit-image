"""
===============================================================================
Harris Corner detector
===============================================================================

The Harris corner filter [1]_ detects "interest points" [2]_ using edge
detection in multiple directions.

.. [1] http://en.wikipedia.org/wiki/Corner_detection
.. [2] http://en.wikipedia.org/wiki/Interest_point_detection
"""

from matplotlib import pyplot as plt

from skimage import data, img_as_float
from skimage.feature import harris


def plot_harris_points(image, filtered_coords):
    """ plots corners found in image"""

    plt.plot()
    plt.imshow(image, cmap=plt.cm.gray)
    plt.plot([p[1] for p in filtered_coords],
             [p[0] for p in filtered_coords],
             'r.')
    plt.axis('off')
    plt.show()

# display results

plt.figure(figsize=(8, 6))
im = img_as_float(data.lena())
im2 = img_as_float(data.text())

filtered_coords = harris(im, min_distance=4)

plt.subplot(121)
plot_harris_points(im, filtered_coords)

filtered_coords = harris(im2, min_distance=4)

plt.subplot(122)
plot_harris_points(im2, filtered_coords)

plt.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9,
                    bottom=0.02, left=0.02, right=0.98)
