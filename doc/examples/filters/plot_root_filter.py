"""
==============================
Root Filtering (Alpha Rooting)
==============================

"""

import matplotlib.pyplot as plt

from skimage import data
from skimage.filters import alpha_rooting


def imshow_side_by_side(original,
                        alpha: float):
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 14))
    filtered = alpha_rooting.alpha_rooting(original, alpha)
    ax[0, 0].imshow(original)
    ax[0, 0].axis('off')
    ax[0, 0].set_title("Original Image and Its Histogram")
    ax[1, 0].hist(original.ravel(), nbins=256)

    ax[0, 1].imshow(filtered)
    ax[0, 1].axis('off')
    ax[0, 1].set_title("Alpha Root Filtered Image and Its Histogram,")
    ax[1, 1].hist(filtered.ravel(), nbins=256)

    plt.show()
###########################################################################
# Contrast enhancement and sharpening properties of alpha-root filtering
# are presented below.


original = data.astronaut()
imshow_side_by_side(original, alpha=0.9)
imshow_side_by_side(original, alpha=0.7)

original = data.camera()
imshow_side_by_side(original, alpha=0.9)
imshow_side_by_side(original, alpha=0.7)

original = data.coffee()
imshow_side_by_side(original, alpha=0.9)
imshow_side_by_side(original, alpha=0.7)
###########################################################################
# For values of alpha that exceed 1.0, we observe blurring.

original = data.astronaut()
imshow_side_by_side(original, alpha=1.5)
imshow_side_by_side(original, alpha=1.7)

original = data.camera()
imshow_side_by_side(original, alpha=1.5)
imshow_side_by_side(original, alpha=1.7)

original = data.coffee()
imshow_side_by_side(original, alpha=1.5)
imshow_side_by_side(original, alpha=1.7)

###########################################################################

# References
# ----------
#
# .. [1] A. K. Jain, Fundamentals of Digital Image Processing.
#        Upper SaddleRiver, NJ: Prentice-Hall, 1989.
