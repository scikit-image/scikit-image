"""
=======================
BRIEF binary descriptor
=======================

This example demonstrates the BRIEF binary description algorithm. The descriptor
consists of relatively few bits and can be computed using a set of intensity
difference tests. The short binary descriptor results in low memory footprint
and very efficient matching based on the Hamming distance metric. BRIEF does not
provide rotation-invariance. Scale-invariance can be achieved by detecting and
extracting features at different scales.

"""

from skimage import data
from skimage import transform
from skimage.feature import (
    match_descriptors,
    corner_peaks,
    corner_harris,
    plot_matched_features,
    BRIEF,
)
from skimage.color import rgb2gray
import matplotlib.pyplot as plt


img1 = rgb2gray(data.astronaut())
tform = transform.AffineTransform(scale=(1.2, 1.2), translation=(0, -100))
img2 = transform.warp(img1, tform)
img3 = transform.rotate(img1, 25)

keypoints1 = corner_peaks(corner_harris(img1), min_distance=5, threshold_rel=0.1)
keypoints2 = corner_peaks(corner_harris(img2), min_distance=5, threshold_rel=0.1)
keypoints3 = corner_peaks(corner_harris(img3), min_distance=5, threshold_rel=0.1)

extractor = BRIEF()

extractor.extract(img1, keypoints1)
keypoints1 = keypoints1[extractor.mask]
descriptors1 = extractor.descriptors

extractor.extract(img2, keypoints2)
keypoints2 = keypoints2[extractor.mask]
descriptors2 = extractor.descriptors

extractor.extract(img3, keypoints3)
keypoints3 = keypoints3[extractor.mask]
descriptors3 = extractor.descriptors

matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
matches13 = match_descriptors(descriptors1, descriptors3, cross_check=True)

fig, ax = plt.subplots(nrows=2, ncols=1)

plt.gray()

plot_matched_features(
    img1,
    img2,
    keypoints0=keypoints1,
    keypoints1=keypoints2,
    matches=matches12,
    ax=ax[0],
)
ax[0].axis('off')
ax[0].set_title("Original Image vs. Transformed Image")

plot_matched_features(
    img1,
    img3,
    keypoints0=keypoints1,
    keypoints1=keypoints3,
    matches=matches13,
    ax=ax[1],
)
ax[1].axis('off')
ax[1].set_title("Original Image vs. Transformed Image")


plt.show()
