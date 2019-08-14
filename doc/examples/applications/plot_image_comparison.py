"""
=======================
Visual image comparison
=======================


"""
import matplotlib.pyplot as plt


from skimage import data, transform, exposure
from skimage.util import compare_images


img1 = data.coins()
img1_equalized = exposure.equalize_hist(img1)
img2 = transform.rotate(img1, 5)


comp_equalized = compare_images(img1, img1_equalized, method='checkerboard')
diff_rotated = compare_images(img1, img2, method='diff')
blend_rotated = compare_images(img1, img2, method='blend')


######################################################################
# Checkerboard
# ============
#
# The checkerboard method alternates tiles from the first and the second
# images.

fig, ax = plt.subplots(ncols=3, figsize=(15, 5))
ax[0].imshow(img1, cmap='gray')
ax[0].set_title('Original')
ax[1].imshow(img1_equalized, cmap='gray')
ax[1].set_title('Equalized')
ax[2].imshow(comp_equalized, cmap='gray')
ax[2].set_title('Checkerboard comparison')
for a in ax:
    a.axis('off')
plt.tight_layout()
plt.plot()

######################################################################
# Diff
# ====
#

plt.imshow(diff_rotated, cmap='gray')
plt.plot()

######################################################################
# Blend
# =====
#

plt.imshow(blend_rotated, cmap='gray')
plt.plot()
