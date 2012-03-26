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
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords],
             [p[0] for p in filtered_coords],
             'b.')
    plt.axis('off')
    plt.show()

# display results
plt.figure(figsize=(9, 3))
im_lena = img_as_float(data.lena())
im_text = img_as_float(data.text())

filtered_coords = harris(im_lena, min_distance=4)

plt.axes([0, 0, 0.3, 0.95])
plot_harris_points(im_lena, filtered_coords)

filtered_coords = harris(im_text, min_distance=4)

plt.axes([0.2, 0, 0.77, 1])
plot_harris_points(im_text, filtered_coords)
