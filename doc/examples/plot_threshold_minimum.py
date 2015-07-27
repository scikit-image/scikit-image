"""
==================================
Minimum Algorithm For Thresholding
==================================

Thresholding is the simplest way to segment objects from a background. If that
background is relatively uniform, then you can use a global threshold value to
binarize the image by pixel-intensity. If there's large variation in the
background intensity, however, adaptive thresholding (a.k.a. local or dynamic
thresholding) may produce better results.

Here, we binarize an image using the `threshold_adaptive` function, which
calculates thresholds in regions of size `block_size` surrounding each pixel
(i.e. local neighborhoods). Each threshold value is the weighted mean of the
local neighborhood minus an offset value.

"""
import matplotlib.pyplot as plt

from skimage import data
from skimage.filters.thresholding import threshold_minimum

image = data.camera()

threshold = threshold_minimum(image)

binarized = image > threshold

fig, axes = plt.subplots(nrows=2, figsize=(7, 8))
ax0, ax1 = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Image')

ax1.imshow(binarized)
ax1.set_title('Thresholded')

for ax in axes:
    ax.axis('off')

plt.show()
