"""
====================
Approximate Polygons
====================

This example shows how to approximate polygonal chains with the Douglas-Peucker
algorithm.
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import ellipse
from skimage.measure import find_contours, approximate_polygon
from skimage.morphology import erosion

img = np.zeros((800, 800), 'int32')

rr, cc = ellipse(250, 250, 180, 230, img.shape)
img[rr, cc] = 1
rr, cc = ellipse(600, 600, 150, 90, img.shape)
img[rr, cc] = 1

plt.gray()
plt.imshow(img)

strel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], 'uint8')
for contour in find_contours(img, 0):
    coords = approximate_polygon(contour, 4)
    plt.plot(coords[:, 1], coords[:, 0], '-r', linewidth=2)
    coords = approximate_polygon(contour, 40)
    plt.plot(coords[:, 1], coords[:, 0], '-g', linewidth=2)

plt.axis((0, 800, 0, 800))
plt.axis('off')
plt.show()
