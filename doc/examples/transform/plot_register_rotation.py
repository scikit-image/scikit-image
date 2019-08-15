"""
===================================
Polar and Log-Polar Transformations
===================================

Rotation differences between two images can be converted to translation
differences along the angular coordinate (:math:`\\theta`) axis of the
polar-transformed images. Scaling differences can be converted to translation
differences along the radial coordinate (:math:`\\rho`) axis if it
is first log transformed (i.e., :math:`\\rho = \ln\sqrt{x^2 + y^2}`). Thus,
in this example, we use phase correlation (``feature.register_translation``)
to recover rotation and scaling differences between two images that share a
center point.
"""

######################################################################
# Recover rotation difference with a polar transform
# ==================================================

import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.draw import circle, set_color
from skimage.feature import register_translation
from skimage.transform import warp_polar, rotate
from skimage.util import img_as_float

radius = 250
angle = 35
image = data.astronaut()
image = img_as_float(image)
rotated = rotate(image, angle)
image_polar = warp_polar(image, radius=radius)
rotated_polar = warp_polar(rotated, radius=radius)

# highlight region to be transformed
rr, cc = circle(256, 256, radius)
set_color(image, (rr, cc), color=(1, 1, 0), alpha=0.2)
set_color(rotated, (rr, cc), color=(1, 1, 0), alpha=0.2)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.ravel()
ax[0].set_title("Original")
ax[0].imshow(image)
ax[1].set_title("Rotated")
ax[1].imshow(rotated)
ax[2].set_title("Polar-Transformed Original")
ax[2].imshow(image_polar)
ax[3].set_title("Polar-Transformed Rotated")
ax[3].imshow(rotated_polar)
plt.show()

shifts, error, phasediff = register_translation(image_polar, rotated_polar)
print(f"Expected value for counterclockwise rotation in degrees: {angle}")
print(f"Recovered value for counterclockwise rotation: {shifts[0]}")

######################################################################
# Recover rotation and scaling differences with log-polar transform
# =================================================================

from skimage.transform import rescale

radius = 250
angle = 53
scale = 2.2
image = data.astronaut()
image = img_as_float(image)
rotated = rotate(image, angle)
rescaled = rescale(rotated, scale, multichannel=True)
image_polar = warp_polar(image, radius=radius, scaling='log')
rescaled_polar = warp_polar(rescaled, radius=radius, scaling='log')

# highlight region to be transformed
im_center_r, im_center_c = (int(x / 2) for x in image.shape[:2])
rr, cc = circle(im_center_r, im_center_c, radius)
set_color(image, (rr, cc), color=(1, 1, 0), alpha=0.2)

res_center_r, res_center_c = (int(x / 2) for x in rescaled.shape[:2])
rr, cc = circle(res_center_r, res_center_c, radius)
set_color(rescaled, (rr, cc), color=(1, 1, 0), alpha=0.2)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.ravel()
ax[0].set_title("Original")
ax[0].imshow(image)
ax[1].set_title("Rotated and Rescaled")
ax[1].imshow(rescaled)
ax[2].set_title("Log-Polar-Transformed Original")
ax[2].imshow(image_polar)
ax[3].set_title("Log-Polar-Transformed Rotated and Rescaled")
ax[3].imshow(rescaled_polar)
plt.show()

shifts, error, phasediff = register_translation(image_polar, rescaled_polar)
shiftr, shiftc = shifts[:2]

# Calculate scale factor from translation
klog = radius / np.log(radius)
shift_scale = 1 / (np.exp(shiftc / klog))

print(f"Expected value for counterclockwise rotation in degrees: {angle}")
print(f"Recovered value for counterclockwise rotation: {shiftr}")
print()
print(f"Expected value for scaling difference: {scale}")
print(f"Recovered value for scaling difference: {shift_scale}")
