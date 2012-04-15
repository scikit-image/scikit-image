"""
============
Fill polygon
============

This example shows how to fill polygons in images.
"""

import matplotlib.pyplot as plt

from skimage.draw import fill_polygon
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
    (220, 490),
    (300, 300),
))
fill_polygon(img, polygon1, color=127)
fill_polygon(img, polygon2, color=255)

plt.gray()
plt.imshow(img)
plt.show()
