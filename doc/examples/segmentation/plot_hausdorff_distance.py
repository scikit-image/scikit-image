"""
==================
Hausdorff Distance
==================

This example shows how to calculate the Hausdorff distance between a
"ground truth" and a "predicted" segmentation mask. The `Hausdorff distance
<https://en.wikipedia.org/wiki/Hausdorff_distance>`__ is the maximum distance
between any point on the contour of the first mask and its nearest point on the
contour of the second mask, and vice-versa.

It can be computed either directly from the segmentation masks using
``hausdorff_distance_mask`` or from contours images (i.e. binary images where
all the pixels on the contours are True) using ``hausdorff_distance``.

In this example, the "contours image" is computed by removing the eroded mask
from the mask itself.
"""
import numpy as np
import matplotlib.pyplot as plt
from skimage import metrics
from skimage.morphology import erosion, disk

# Creates a "ground truth" binary mask with a disk, and a partially overlapping "predicted" rectangle
ground_truth = np.zeros((100, 100), dtype=bool)
predicted = ground_truth.copy()

ground_truth[30:71, 30:71] = disk(20)
predicted[25:65, 40:70] = True

# Creates "contours" image by xor-ing an erosion
se = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
gt_contour = ground_truth ^ erosion(ground_truth, se)
predicted_contour = predicted ^ erosion(predicted, se)

###############################################################################
# Compute and display the Hausdorff distance and the corresponding pair of
# points from the "contours image":

distance = metrics.hausdorff_distance(gt_contour, predicted_contour)
pair = metrics.hausdorff_pair(gt_contour, predicted_contour)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(ground_truth)
plt.subplot(1, 3, 2)
plt.imshow(predicted)
plt.subplot(1, 3, 3)
plt.imshow(gt_contour)
plt.imshow(predicted_contour, alpha=0.5)
plt.plot([pair[0][1], pair[1][1]], [pair[0][0], pair[1][0]], 'wo-')
plt.title(f"HD={distance:.3f}px")
plt.show()

###############################################################################
# Compute and display the Hausdorff distance and the corresponding pair of
# points from the segmentation masks directly:#

distance = metrics.hausdorff_distance_mask(ground_truth, predicted)
pair = metrics.hausdorff_pair_mask(ground_truth, predicted)

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(ground_truth)
plt.subplot(1, 3, 2)
plt.imshow(predicted)
plt.subplot(1, 3, 3)
plt.imshow(ground_truth)
plt.imshow(predicted, alpha=0.5)
plt.plot([pair[0][1], pair[1][1]], [pair[0][0], pair[1][0]], 'wo-')
plt.title(f"HD={distance:.3f}px")
plt.show()
