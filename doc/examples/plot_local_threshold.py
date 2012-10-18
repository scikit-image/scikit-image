"""
=====================
Local Thresholding
=====================

Thresholding is the simplest way to segment objects from a background. If that
background is relatively uniform, then you can use a global threshold value to
binarize the image by pixel-intensity. If there's large variation in the
background intensity, however, adaptive thresholding (a.k.a. local or dynamic
thresholding) may produce better results.

Here, we binarize an image using the `threshold_adaptive` function, which
calculates thresholds in regions of size `block_size` surrounding each pixel
(i.e. local neighborhoods). Each threshold value is the weighted mean of the
local neighborhood minus an offset value.

An other approach is to binarize locally the image using local histogram distribution.

rank.threshold function set pixels higher than the local mean to 1, to 0 otherwize
rank.morph_contr_enh replaces each pixel by the local minimum (or local maximum) if the
pixel gray level is more close to the local minimum (resp. by the local maximum
if the pixel gray level is more close to the local maximum).

"""
import matplotlib.pyplot as plt

from skimage import data
from skimage.filter import threshold_otsu, threshold_adaptive

from skimage.filter.rank import threshold,morph_contr_enh
from skimage.morphology import disk


image = data.page()

global_thresh = threshold_otsu(image)
binary_global = image > global_thresh

block_size = 40
binary_adaptive = threshold_adaptive(image, block_size, offset=10)

selem = disk(20)
loc_thresh = threshold(image,selem=selem)
loc_morph_contr_enh = morph_contr_enh(image,selem=selem)

fig, axes = plt.subplots(nrows=5, figsize=(7, 8))
ax0, ax1, ax2, ax3, ax4 = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Image')

ax1.imshow(binary_global)
ax1.set_title('Global thresholding')

ax2.imshow(binary_adaptive)
ax2.set_title('Adaptive thresholding')

ax3.imshow(loc_thresh)
ax3.set_title('Local thresholding')

ax4.imshow(loc_morph_contr_enh)
ax4.set_title('Local morphological contrast enhancement')


for ax in axes:
    ax.axis('off')

plt.show()
