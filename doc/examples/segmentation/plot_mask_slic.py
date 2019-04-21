"""
======================
maskSLIC Demonstration
======================

This example is about comparing the segmentations obtained using the
plain SLIC method [1]_ and its masked version maskSLIC [2]_.


.. [1] Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi,
    Pascal Fua, and Sabine Suesstrunk, SLIC Superpixels Compared to
    State-of-the-art Superpixel Methods, TPAMI, May 2012.

.. [2] Irving, Benjamin. "maskSLIC: regional superpixel generation
    with application to local pathology characterisation in medical
    images." arXiv preprint arXiv:1606.09518 (2016).

"""

import matplotlib.pyplot as plt

from skimage import data
from skimage import color
from skimage import morphology
from skimage import segmentation

# Input data
img = data.immunohistochemistry()

# Compute a mask
lum = color.rgb2gray(img)
mask = morphology.remove_small_holes(
    morphology.remove_small_objects(
        lum < 0.7, 500),
    500)

mask = morphology.opening(mask, morphology.disk(3))

# SLIC result
slic = segmentation.slic(img, n_segments=200)

# maskSLIC result
m_slic = segmentation.slic(img, n_segments=100, mask=mask)

# Display result
fig = plt.figure(figsize=(10, 10))
ax1, ax2, ax3, ax4 = fig.subplots(2, 2, sharex=True, sharey=True).ravel()

ax1.imshow(img)
ax1.set_axis_off()
ax1.set_title("Origin image")

ax2.imshow(mask, cmap="gray")
ax2.set_axis_off()
ax2.set_title("Considered mask")

ax3.imshow(segmentation.mark_boundaries(img, slic))
ax3.contour(mask, colors='red', linewidths=1)
ax3.set_axis_off()
ax3.set_title("SLIC")

ax4.imshow(segmentation.mark_boundaries(img, m_slic))
ax4.contour(mask, colors='red', linewidths=1)
ax4.set_axis_off()
ax4.set_title("maskSLIC")

fig.set_tight_layout(True)

plt.show()
