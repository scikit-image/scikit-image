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
lfw_images = data.lfw_subset()
for i in range(20):
    ax[i].imshow(lfw_images[90 + i], cmap=plt.cm.gray)
    ax[i].axis("off")
fig.tight_layout()
plt.show()


from matplotlib.offsetbox import AnchoredText

ax0 = plt.subplot(2, 2, 1)
ax0.imshow(cylce_images[0])
ax0.add_artist(
    AnchoredText(
        "Stereo",
        prop=dict(size=20),
        frameon=True,
        borderpad=0,
        loc="upper left",
    )
)
ax1 = plt.subplot(2, 2, 2)
ax1.imshow(vortex_images[0])
ax1.add_artist(
    AnchoredText(
        "PIV",
        prop=dict(size=20),
        frameon=True,
        borderpad=0,
        loc="upper left",
    )
)

ax2 = plt.subplot(2, 4, 5)
ax2.imshow(lfw_images[90 + 1], cmap="gray")
ax3 = plt.subplot(2, 4, 6)
ax3.imshow(lfw_images[90 + 2], cmap="gray")
ax4 = plt.subplot(2, 4, 7)
ax4.imshow(lfw_images[90 + 3], cmap="gray")
ax5 = plt.subplot(2, 4, 8)
ax5.imshow(lfw_images[90 + 4], cmap="gray")
for ax in [ax0, ax1, ax2, ax3, ax4, ax5]:
    ax.axis("off")
plt.tight_layout()
plt.show()
