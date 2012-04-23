"""
===========
Fill shapes
===========

This example shows how to fill several different shapes:
* line
* polygon
* circle
* ellipse

"""

import matplotlib.pyplot as plt

from skimage.draw import line, polygon, circle, ellipse
import numpy as np


img = np.zeros((500, 500, 3), 'uint8')

#: draw line
rr, cc = line(120, 123, 20, 400)
img[rr,cc,0] = 255

#: fill polygon
poly = np.array((
    (300, 300),
    (480, 320),
    (380, 430),
    (220, 590),
    (300, 300),
))
rr, cc = polygon(poly[:,0], poly[:,1], img.shape)
img[rr,cc,1] = 255

#: fill circle
rr, cc = circle(200, 200, 100, img.shape)
img[rr,cc,:] = (255, 255, 0)

#: fill ellipse
rr, cc = ellipse(300, 300, 100, 200, img.shape)
img[rr,cc,2] = 255

plt.imshow(img)
plt.show()