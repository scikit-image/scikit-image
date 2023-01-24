"""
===============================================
Separate colors in immunohistochemical staining
===============================================

Color deconvolution consists in the separation of features by their colors.

In this example we separate the immunohistochemical (IHC) staining from the
hematoxylin counterstaining. The separation is achieved with the method
described in [1]_ and known as "color deconvolution".

The IHC staining expression of the FHL2 protein is here revealed with
diaminobenzidine (DAB) which gives a brown color.


.. [1] A. C. Ruifrok and D. A. Johnston, "Quantification of histochemical
       staining by color deconvolution," Analytical and quantitative
       cytology and histology / the International Academy of Cytology [and]
       American Society of Cytology, vol. 23, no. 4, pp. 291-9, Aug. 2001.
       PMID: 11531144

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.color import separate_stains, combine_stains, stain_color_matrix

# Example IHC image
ihc_rgb = data.immunohistochemistry()

# Set the clear glass to pure white, which we measured by looking at the whitest
# region of the image.
ihc_rgb = ihc_rgb / [236, 236, 234]

# Stain color matrix
m = stain_color_matrix(('Hematoxylin', 'DAB'))

# Separate the stains from the IHC image
ihc_hd = separate_stains(ihc_rgb, m)

# Create an RGB image for each of the stains
null = np.zeros_like(ihc_hd[:, :, 0])
ihc_h = combine_stains(np.stack((ihc_hd[:, :, 0], null), axis=-1), m)
ihc_d = combine_stains(np.stack((null, ihc_hd[:, :, 1]), axis=-1), m)

# Display
fig, axes = plt.subplots(2, 2, figsize=(7, 6), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(ihc_rgb)
ax[0].set_title("Original image")

ax[1].imshow(ihc_h)
ax[1].set_title("Hematoxylin")

ax[2].imshow(ihc_d)
ax[2].set_title("DAB")

for a in ax.ravel():
    a.axis('off')

fig.tight_layout()


######################################################################
# Now we can easily manipulate the hematoxylin and DAB channels:

from skimage.exposure import rescale_intensity

# Rescale hematoxylin and DAB channels and give them a fluorescence look
h = rescale_intensity(ihc_hd[:, :, 0], out_range=(0, 1),
                      in_range=(0, np.percentile(ihc_hd[:, :, 0], 99)))
d = rescale_intensity(ihc_hd[:, :, 1], out_range=(0, 1),
                      in_range=(0, np.percentile(ihc_hd[:, :, 1], 99)))

# Cast the two channels into an RGB image, as the blue and green channels
# respectively
zdh = np.dstack((null, d, h))

fig = plt.figure()
axis = plt.subplot(1, 1, 1, sharex=ax[0], sharey=ax[0])
axis.imshow(zdh)
axis.set_title('Stain-separated image (rescaled)')
axis.axis('off')
plt.show()
