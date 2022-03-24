"""
===============
Specific images
===============

"""
import matplotlib.pyplot as plt
import matplotlib

from skimage import data

matplotlib.rcParams["font.size"] = 18

######################################################################
#
# Stereo images
# =============


fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

cylce_images = data.stereo_motorcycle()
ax[0].imshow(cylce_images[0])
ax[1].imshow(cylce_images[1])

fig.tight_layout()
plt.show()


######################################################################
#
# PIV images
# =============


fig, axes = plt.subplots(1, 2, figsize=(8, 4))
ax = axes.ravel()

vortex_images = data.vortex()
ax[0].imshow(vortex_images[0])
ax[1].imshow(vortex_images[1])

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
lfs_images = data.lfw_subset()
for i in range(20):
    ax[i].imshow(lfs_images[90+i], cmap=plt.cm.gray)
    ax[i].axis('off')
fig.tight_layout()
plt.show()




############################################################################
# Thumbnail image for the gallery

# sphinx_gallery_thumbnail_number = -1
from matplotlib.offsetbox import AnchoredText

ax1 = plt.subplot(2, 2, 1)
ax1.imshow(cylce_images[0])
ax1.add_artist(
    AnchoredText(
        "Stereo", prop=dict(size=20), frameon=True, borderpad=0, loc="upper left"
    )
)
ax2 = plt.subplot(2, 2, 2)
ax2.imshow(vortex_images[0])
ax2.add_artist(
    AnchoredText("PIV", prop=dict(size=20), frameon=True, borderpad=0, loc="upper left")
)

ax3 = plt.subplot(2, 4, 5)
ax3.imshow(lfs_images[90+1], cmap="gray")
ax4 = plt.subplot(2, 4, 6)
ax4.imshow(lfs_images[90+2], cmap="gray")
ax5 = plt.subplot(2, 4, 7)
ax5.imshow(lfs_images[90+3], cmap="gray")
ax6 = plt.subplot(2, 4, 8)
ax6.imshow(lfs_images[90+4], cmap="gray")
for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
    ax.axis("off")
plt.tight_layout()