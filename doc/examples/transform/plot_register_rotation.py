r"""
===================================
Polar and Log-Polar Transformations
===================================

Rotation differences between two images can be converted to translation
differences along the angular coordinate (:math:`\theta`) axis of the
polar-transformed images. Scaling differences can be converted to translation
differences along the radial coordinate (:math:`\rho`) axis if it
is first log transformed (i.e., :math:`\rho = \ln\sqrt{x^2 + y^2}`). Thus,
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
from skimage.feature import register_translation
from skimage.transform import warp_polar, rotate
from skimage.util import img_as_float

radius = 705
angle = 35
image = data.retina()
image = img_as_float(image)
rotated = rotate(image, angle)
image_polar = warp_polar(image, radius=radius, multichannel=True)
rotated_polar = warp_polar(rotated, radius=radius, multichannel=True)

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
print("Expected value for counterclockwise rotation in degrees: "
      f"{angle}")
print("Recovered value for counterclockwise rotation: "
      f"{shifts[0]}")

######################################################################
# Recover rotation and scaling differences with log-polar transform
# =================================================================

from skimage.transform import rescale

# radius must be large enough to capture useful info in larger image
radius = 1500
angle = 53.7
scale = 2.2
image = data.retina()
image = img_as_float(image)
rotated = rotate(image, angle)
rescaled = rescale(rotated, scale, multichannel=True)
image_polar = warp_polar(image, radius=radius,
                         scaling='log', multichannel=True)
rescaled_polar = warp_polar(rescaled, radius=radius,
                            scaling='log', multichannel=True)

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

# setting `upsample_factor` can increase precision
tparams = register_translation(image_polar, rescaled_polar, upsample_factor=20)
shifts, error, phasediff = tparams
shiftr, shiftc = shifts[:2]

# Calculate scale factor from translation
klog = image_polar.shape[1] / np.log(radius)
shift_scale = 1 / (np.exp(shiftc / klog))

print(f"Expected value for cc rotation in degrees: {angle}")
print(f"Recovered value for cc rotation: {shiftr}")
print()
print(f"Expected value for scaling difference: {scale}")
print(f"Recovered value for scaling difference: {shift_scale}")

######################################################################
# Register rotation and scaling on a translated image
# =================================================================
#
# The above examples only work when the images to be registered share a
# center. However, it is more often the case that there is also a translation
# component to the difference between two images to be registered. One
# approach to register rotation, scaling and translation is to first correct
# for rotation and scaling, then solve for translation. It is possible to
# resolve rotation and scaling differences for translated images by working on
# the magnitude spectra of the fourier transformed images.

from skimage.color import rgb2gray
from skimage import draw, filters
from scipy.fftpack import fft2, fftshift

angle = 36
scale = 1.4
shiftr = 30
shiftc = 15


def create_image_pair(image, angle, scale, shiftr, shiftc):
    translated = image[shiftr:, shiftc:]
    rotated = rotate(translated, angle)
    rescaled = rescale(rotated, scale)
    shaper, shapec = image.shape
    rts_image = rescaled[:shaper, :shapec]
    return image, rts_image


image = rgb2gray(data.retina())
image, rts_image = create_image_pair(image, angle=angle, scale=scale,
                                     shiftr=shiftr, shiftc=shiftc)

warped_image = warp_polar(image, scaling="log")
warped_rts = warp_polar(rts_image, scaling="log")

# When center is not shared, log-polar transform is not helpful!
fig, axes = plt.subplots(2, 2, figsize=(8, 8))
ax = axes.ravel()
ax[0].set_title("Original Image")
ax[0].imshow(image)
ax[1].set_title("Modified Image")
ax[1].imshow(rts_image)
ax[2].set_title("Log-Polar-Transformed Original")
ax[2].imshow(warped_image)
ax[3].set_title("Log-Polar-Transformed Modified")
ax[3].imshow(warped_rts)
plt.show()


# Use difference of gaussians to enhance image features
def dog(image, sigma1, sigma2):
    image = filters.gaussian(image, sigma1) - filters.gaussian(image, sigma2)
    return image


image = dog(image, 5, 20)
rts_image = dog(rts_image, 5, 20)


# Window the images and take the magnitude of the FFT
def window_image(image, window_diameter=0.8, window_decay=10):
    """window_diameter is relative to length of shortest axis
    window_decay determines steepness of window edges (higher is steeper)"""
    window_center = np.divide(image.shape, 2)
    radius = (window_diameter * np.min(image.shape)) / 2
    window = np.zeros_like(image)
    rr, cc = draw.circle(window_center[0], window_center[1], radius)
    window[rr, cc] = 1
    window = filters.gaussian(window, sigma=(radius / window_decay))
    return image * window


image = window_image(image)
rts_image = window_image(rts_image)
image_fs = np.abs(fftshift(fft2(image)))
rts_fs = np.abs(fftshift(fft2(rts_image)))


# Create log-polar transformed images to register
shape = image_fs.shape
radius = shape[0] / 8
warped_image_fs = warp_polar(image_fs, radius=radius, output_shape=shape,
                             scaling='log', order=0)
warped_rts_fs = warp_polar(rts_fs, radius=radius, output_shape=shape,
                           scaling='log', order=0)

warped_image_fs = warped_image_fs[:int(shape[0]/2), :]
warped_rts_fs = warped_rts_fs[:int(shape[0]/2), :]
tparams = register_translation(warped_image_fs, warped_rts_fs,
                               upsample_factor=10)

# Use translation parameters to calculate rotation and scaling parameters
shifts, error, phasediff = tparams
shiftr, shiftc = shifts[:2]
recovered_angle = (360 / shape[0]) * shiftr

klog = shape[1] / np.log(radius)
shift_scale = np.exp(shiftc / klog)

print(recovered_angle, shift_scale)
print(f"Expected value for cc rotation in degrees: {angle}")
print(f"Recovered value for cc rotation: {recovered_angle}")
print()
print(f"Expected value for scaling difference: {scale}")
print(f"Recovered value for scaling difference: {shift_scale}")
