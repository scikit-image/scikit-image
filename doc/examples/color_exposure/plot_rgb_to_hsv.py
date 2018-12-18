"""
================
RGB to HSV
================

This example converts an image with RGB channels into an image with HSV (Hue,
Saturation,Value) channels.

This coordinate system is useful for color-based thresholding since the color
of a pixel is only described by its Hue value (and not the R,G,B values). [1]_

References
----------
.. [1] https://en.wikipedia.org/wiki/HSL_and_HSV

"""
import matplotlib.pyplot as plt

from skimage import data
from skimage.color import rgb2hsv

original = data.coffee()
hsv_img = rgb2hsv(original)
hue_img = hsv_img[:,:,0]

global_thresh = 0.04
binary_global = hue_img > global_thresh

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.ravel()

ax[0].imshow(original)
ax[0].set_title("Original")
ax[1].imshow(hsv_img)
ax[1].set_title("HSV image")
ax[2].hist(hue_img.ravel(),512)
ax[2].set_title("Histogram of Hue values with threshold")
ax[2].axvline(x=global_thresh, color='r', linestyle='dashed', linewidth=2)
ax[2].set_xbound(0,0.12)
ax[3].imshow(binary_global)
ax[3].set_title("Binary image")

fig.tight_layout()
plt.show()