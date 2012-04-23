"""
============
Fill polygon
============

This example shows how to fill polygons in images.
"""

import matplotlib.pyplot as plt

from skimage.draw import polygon
import numpy as np


img = np.zeros((500, 500), 'uint8')
polygon1 = np.array((
    (50, 50),
    (150, 30),
    (400, 100),
    (300, 200),
    (480, 400),
    (100, 420),
    (50, 50),
))
polygon2 = np.array((
    (300, 300),
    (480, 320),
    (380, 430),
    (220, 590),
    (300, 300),
))
rr, cc = polygon(polygon1, img.shape)
img[rr,cc] = 127
rr, cc = polygon(polygon2, img.shape)
img[rr,cc] = 255

plt.gray()
plt.imshow(img)
plt.show()