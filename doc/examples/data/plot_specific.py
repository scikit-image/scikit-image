"""
===============
Specific images
===============

"""
import matplotlib.pyplot as plt
import matplotlib

from skimage import data

matplotlib.rcParams['font.size'] = 18

######################################################################
#
# Stereo images
# =============


fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

images = data.stereo_motorcycle()
ax[0].imshow(images[0])
ax[1].imshow(images[1])

fig.tight_layout()
plt.show()

######################################################################
#
# Faces and non-faces dataset
# ===========================
#
# A sample of 20 over 200 images is displayed.


fig, axes = plt.subplots(4, 5, figsize=(20, 20))
ax = axes.ravel()
images = data.lfw_subset()
for i in range(20):
    ax[i].imshow(images[90+i], cmap=plt.cm.gray)
    ax[i].axis('off')
fig.tight_layout()
plt.show()

######################################################################
#
# 3D mouse brain images
# =====================
#
# The Allen Institute's mouse brain atlas and a cleared mouse brain are
# displayed. The two columns respectively display the allen_mouse_brain_atlas
# and a cleared_mouse_brain, where each row shows a central slice along that
# dimension.

fig, axes = plt.subplots(3, 2, figsize=(8, 12))

standard_brain = data.allen_mouse_brain_atlas()
specific_brain = data.cleared_mouse_brain()
images = standard_brain, specific_brain

# Plot the center slice of each image in each dimension.
for image_index, image in enumerate(images):
    for dim in range(3):
        ax = axes[dim, image_index]
        ax.imshow(image.take(image.shape[dim] // 2, axis=dim))
        ax.axis('off')
        
fig.tight_layout()
plt.show()
