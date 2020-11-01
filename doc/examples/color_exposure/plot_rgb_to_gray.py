"""
================
RGB to grayscale
================

This example converts an image with RGB channels into an image with a single
grayscale channel.

The value of each grayscale pixel is calculated as the weighted sum of the
corresponding red, green and blue pixels as::

        Y = 0.2125 R + 0.7154 G + 0.0721 B

These weights are used by CRT phosphors as they better represent human
perception of red, green and blue than equal weights. [1]_

References
----------
.. [1] http://poynton.ca/PDFs/ColorFAQ.pdf

"""
import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2gray

original = data.astronaut()
grayscale = rgb2gray(original)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

ax[0].imshow(original)
ax[0].set_title("Original")
ax[1].imshow(grayscale, cmap=plt.cm.gray)
ax[1].set_title("Grayscale")

fig.tight_layout()
plt.show()
