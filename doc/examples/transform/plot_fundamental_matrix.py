"""
=============================
Fundamental matrix estimation
=============================

This example demonstrates how to robustly estimate epipolar geometry between two
views using sparse ORB feature correspondences.

The fundamental matrix relates corresponding points between a pair of
uncalibrated images. The matrix transforms homogeneous image points in one image
to epipolar lines in the other image.

Uncalibrated means that the intrinsic calibration (focal lengths, pixel skew,
principal point) of the two cameras is not known. The fundamental matrix thus
enables projective 3D reconstruction of the captured scene. If the calibration
is known, estimating the essential matrix enables metric 3D reconstruction of
the captured scene.

"""
from skimage import data
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, ORB, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform
import matplotlib.pyplot as plt

img1, img2 = map(rgb2gray, data.stereo_motorcycle())

descriptor_extractor = ORB()

descriptor_extractor.detect_and_extract(img1)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(img2)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors

matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)

model, inliers = ransac((keypoints1[matches12[:, 0]],
                         keypoints2[matches12[:, 1]]),
                        FundamentalMatrixTransform, min_samples=8,
                        residual_threshold=2, max_trials=5000)

print("Number of matches:", matches12.shape[0])
print("Number of inliers:", inliers.sum())

fig, ax = plt.subplots(nrows=1, ncols=1)

plt.gray()

plot_matches(ax, img1, img2, keypoints1, keypoints2, matches12[inliers],
             only_matches=True)
ax.axis('off')

plt.show()
