"""
============
Seam Carving
============

This example demonstrates how images can be resized using seam carving [1]_.
Resizing to a new aspect ratio distorts image contents. Seam carving attempts
to resize *without* distortion, by removing regions of an image which are less
important. In this example we are using the Sobel filter to signify the
importance of each pixel.

.. [1] Shai Avidan and Ariel Shamir
       "Seam Carving for Content-Aware Image Resizing"
       http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Avidan07.pdf

"""
from skimage import data, draw
from skimage import transform, util
import numpy as np
from skimage import filters, color
from matplotlib import pyplot as plt


hl_color = np.array([0, 1, 0])

img = data.rocket()
img = util.img_as_float(img)
eimg = filters.sobel(color.rgb2gray(img))

plt.title('Original Image')
plt.imshow(img)

"""
.. image:: PLOT2RST.current_figure
"""

resized = transform.resize(img, (img.shape[0], img.shape[1] - 200))
plt.figure()
plt.title('Resized Image')
plt.imshow(resized)


"""
.. image:: PLOT2RST.current_figure
"""

out = transform.seam_carve(img, eimg, 'vertical', 200)
plt.figure()
plt.title('Resized using Seam Carving')
plt.imshow(out)

"""
.. image:: PLOT2RST.current_figure

As you can see, resizing has distorted the rocket and the objects around,
whereas seam carving has resized by removing the empty spaces in between.

Object Removal
--------------

Seam carving can also be used to remove artifacts from images. To do that, we
have to ensure that pixels to be removed get less importance. In the following
code I approximately mark the rocket with a mask, and then decrease the
importance of those pixels.

"""

masked_img = img.copy()

poly = [(404, 281), (404, 360), (359, 364), (338, 337), (145, 337), (120, 322),
        (145, 304), (340, 306), (362, 284)]
pr = np.array([p[0] for p in poly])
pc = np.array([p[1] for p in poly])
rr, cc = draw.polygon(pr, pc)

masked_img[rr, cc, :] = masked_img[rr, cc, :]*0.5 + hl_color*.5
plt.figure()
plt.title('Object Marked')

plt.imshow(masked_img)
"""
.. image:: PLOT2RST.current_figure
"""

eimg[rr, cc] -= 1000

plt.figure()
plt.title('Object Removed')
out = transform.seam_carve(img, eimg, 'vertical', 90)
resized = transform.resize(img, out.shape)
plt.imshow(out)
plt.show()
"""
.. image:: PLOT2RST.current_figure
"""
