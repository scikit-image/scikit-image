"""
========================================
Circular and Elliptical Hough Transforms
========================================

The Hough transform in its simplest form is a `method to detect
straight lines <http://en.wikipedia.org/wiki/Hough_transform>`__
but it can also be used to detect circles or ellipses.
The algorithm assumes that the edge is detected and it is robust against
noise or missing points.

Circle detection
================

In the following example, the Hough transform is used to detect
coin positions and match their edges. We provide a range of
plausible radii. For each radius, two circles are extracted and
we finally keep the five most prominent candidates.
The result shows that coin positions are well-detected.


Algorithm overview
------------------

Given a black circle on a white background, we first guess its
radius (or a range of radii) to construct a new circle.
This circle is applied on each black pixel of the original picture
and the coordinates of this circle are voting in an accumulator.
From this geometrical construction, the original circle center
position receives the highest score.

Note that the accumulator size is built to be larger than the
original picture in order to detect centers outside the frame.
Its size is extended by two times the larger radius.

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, filter, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte


# Load picture and detect edges
image = img_as_ubyte(data.coins()[0:95, 70:370])
edges = filter.canny(image, sigma=3, low_threshold=10, high_threshold=50)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))

# Detect two radii
hough_radii = np.arange(15, 30, 2)
hough_res = hough_circle(edges, hough_radii)

centers = []
accums = []
radii = []

for radius, h in zip(hough_radii, hough_res):
    # For each radius, extract two circles
    peaks = peak_local_max(h, num_peaks=2)
    centers.extend(peaks)
    accums.extend(h[peaks[:, 0], peaks[:, 1]])
    radii.extend([radius, radius])

# Draw the most prominent 5 circles
image = color.gray2rgb(image)
for idx in np.argsort(accums)[::-1][:5]:
    center_x, center_y = centers[idx]
    radius = radii[idx]
    cx, cy = circle_perimeter(center_y, center_x, radius)
    image[cy, cx] = (220, 20, 20)

ax.imshow(image, cmap=plt.cm.gray)
plt.show()


"""
Ellipse detection
=================

In this second example, the aim is to detect the edge of a coffee cup.
Basically, this is a projection of a circle, i.e. an ellipse.
The problem to solve is much more difficult because five parameters have to be
determined, instead of three for circles.


Algorithm overview
------------------

The algorithm takes two different points belonging to the ellipse. It assumes
that it is the main axis. A loop on all the other points determines how much
an ellipse passes to them. A good match corresponds to high accumulator values.

A full description of the algorithm can be found in reference [1]_.


References
----------
.. [1] Xie, Yonghong, and Qiang Ji. "A new efficient ellipse detection
       method." Pattern Recognition, 2002. Proceedings. 16th International
       Conference on. Vol. 2. IEEE, 2002
"""
import matplotlib.pyplot as plt

from skimage import data, filter, color
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

# Load picture, convert to grayscale and detect edges
image_rgb = data.load('coffee.png')[0:220, 100:450]
image_gray = color.rgb2gray(image_rgb)
edges = filter.canny(image_gray, sigma=2.0,
                     low_threshold=0.55, high_threshold=0.8)

# Perform a Hough Transform
# The accuracy corresponds to the bin size of a major axis.
# The value is chosen in order to get a single high accumulator.
# The threshold eliminates low accumulators
accum = hough_ellipse(edges, accuracy=10, threshold=170, min_size=50)
accum.sort(key=lambda x: x[0])
best = accum[-1]
# Estimated parameters for the ellipse
center_y = int(best[1])
center_x = int(best[2])
yradius = int(best[3])
xradius = int(best[4])
angle = best[5]

# Draw the ellipse on the original image
cy, cx = ellipse_perimeter(center_y, center_x,
                           yradius, xradius, orientation=angle)
image_rgb[cy, cx] = (0, 0, 1)
# Draw the edge (white) and the resulting ellipse (red)
edges = color.gray2rgb(edges)
edges[cy, cx] = (250, 0, 0)

fig = plt.subplots(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.title('Original picture')
plt.imshow(image_rgb)
plt.subplot(1, 2, 2)
plt.title('Edge (white) and result (red)')
plt.imshow(edges)

plt.show()
