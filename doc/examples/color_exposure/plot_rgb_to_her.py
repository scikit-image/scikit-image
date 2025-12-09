"""
===============================================
Separate stains in hematoxylin and eosin (H&E) stain images
===============================================

Color deconvolution consists in the separation of features by their colors.

In this example we separate hematoxylin and eosin stains by using color deconvolution as described
by the authors of Qupath[2].

Hematoxylin is a blue/purple stain which stains cell nuclei, and eosin is a pink stain which stains
the extracellular matrix and cytoplasm pink. The "residual" channel here is for other colors sometimes
found in HE stained slides, like yellow or green due to various artefacts or debris.

In the image below, we can see the yellow color in the original image highlighted on in the residual channel.

It is important to note that the authors of QuPath do not recommend using these color deconvolved
stain channels for quantitative interpretation [3]. The function described here uses default
stain vector values for hematoxylin and eosin, whereas in practice, there is high variability in
stain colors between different institutions and settings. QuPath recommends recalculating stain vectors for
your own specific images, which is not performed by this library.

.. [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
       staining by color deconvolution," Analytical and quantitative
       cytology and histology / the International Academy of Cytology [and]
       American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.
       PMID: 11531144

.. [2] Bankhead, P. et al. QuPath: Open source software for digital pathology
        image analysis. Scientific Reports (2017). https://doi.org/10.1038/s41598-017-17204-5

.. [3] https://qupath.readthedocs.io/en/stable/docs/tutorials/separating_stains.html#brightfield-images
"""

from skimage import data
from skimage.color import rgb2her, her2rgb
import matplotlib.pyplot as plt
import numpy as np

# Load HE image of skin histology from skimage.data
img = data.skin()

img_her = rgb2her(img)
# Create an RGB image for each of the stains
null = np.zeros_like(img_her[:, :, 0])
img_h = her2rgb(np.stack((img_her[:, :, 0], null, null), axis=-1))
img_e = her2rgb(np.stack((null, img_her[:, :, 1], null), axis=-1))
img_r = her2rgb(np.stack((null, null, img_her[:, :, 2]), axis=-1))

# Display
fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(img)
ax[0].set_title("Original image")

ax[1].imshow(img_h)
ax[1].set_title("Hematoxylin")

ax[2].imshow(img_e)
ax[2].set_title("Eosin")

ax[3].imshow(img_r)
ax[3].set_title("Residual")  # Residual should be close to empty for HE images
# we can see some yellow/green staining in the bottom left

for a in ax.ravel():
    a.axis('off')

fig.tight_layout()

######################################################################
# Now we can easily manipulate the hematoxylin and eosin channels:

from skimage.exposure import rescale_intensity

# Rescale hematoxylin and eosin channels and give them a fluorescence look
h = rescale_intensity(
    img_her[:, :, 0],
    out_range=(0, 1),
    in_range=(0, np.percentile(img_her[:, :, 0], 99)),
)
e = rescale_intensity(
    img_her[:, :, 1],
    out_range=(0, 1),
    in_range=(0, np.percentile(img_her[:, :, 1], 99)),
)

# Cast the two channels into an RGB image, as the blue and green channels
# respectively
zdh = np.dstack((null, e, h))

fig = plt.figure()
axis = plt.subplot(1, 1, 1, sharex=ax[0], sharey=ax[0])
axis.imshow(zdh)
axis.set_title('Stain-separated image (rescaled)')
axis.axis('off')
plt.show()
