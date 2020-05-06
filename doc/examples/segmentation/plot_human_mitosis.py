"""
================================
Segment human cells (in mitosis)
================================

In this example, we analyze a microscopy image of human cells. We use data
provided by Jason Moffat [1]_ through [CellProfiler](https://cellprofiler.org/examples/#human-cells).

.. [1] Moffat J, Grueneberg DA, Yang X, Kim SY, Kloepfer AM, Hinkle G, Piqani B, Eisenhaure TM, Luo B, Grenier JK, Carpenter AE, Foo SY, Stewart SA, Stockwell BR, Hacohen N, Hahn WC, Lander ES, Sabatini DM, Root DE (2006) "A lentiviral RNAi library for human and mouse genes applied to an arrayed viral high-content screen" Cell, 124(6):1283-98. DOI: [10.1016/j.cell.2006.01.040](https://doi.org/10.1016/j.cell.2006.01.040). PMID: 16564017

"""

import matplotlib.pyplot as plt
import numpy as np

from skimage import filters, io, morphology, util


image = io.imread('https://github.com/CellProfiler/examples/blob/master/ExampleHuman/images/AS_09125_050116030001_D03f00d0.tif?raw=true')

#####################################################################
# This image is a TIFF file. If you run into issues loading it, please
# consider using ``external.tifffile.imread`` or following:
# https://github.com/scikit-image/scikit-image/issues/4326#issuecomment-559595147

fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
plt.show()

#####################################################################
# We can see many cells on a dark background. They are fairly smooth and
# elliptical. In addition, some of them present brighter spots: They are going
# through the process of cell division
# ([mitosis](https://en.wikipedia.org/wiki/Mitosis)).

#####################################################################
# Estimate the mitotic index
# ==========================
# Cellular biology uses the mitotic index to quantify cell division (and,
# hence, cell population growth). By definition, it is the ratio of cells in
# mitosis (over the total number of cells). To analyze the above image,
# we are thus interested in two thresholds: one separating the cells from the
# background, the other separating the dividing nuclei (brighter spots) from
# the cytoplasm of their respective mother cells (and, intensity-wise, from
# the other cells). To separate these three different classes of pixels, we
# resort to :ref:`sphx-glr-auto-examples-segmentation-plot-multiotsu-py`.

thresholds = filters.threshold_multiotsu(image)
regions = np.digitize(image, bins=thresholds)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
ax[0].imshow(image)
ax[0].set_title('Original')
ax[0].axis('off')
ax[1].imshow(regions)
ax[1].set_title('Multi-Otsu thresholding')
ax[1].axis('off')
plt.show()

#####################################################################
# Since there are touching cells, thresholding is not enough to segment all
# the cells. If it were, we could readily compute a mitotic index for this
# sample:

cells = image > thresholds[0]
dividing =  image > thresholds[1]
labeled_cells = morphology.label(cells)
labeled_dividing = morphology.label(dividing)
naive_mi = labeled_dividing.max() / labeled_cells.max()

#####################################################################
# Whoa, this can't be!

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
ax[0].imshow(image)
ax[0].set_title('Original')
ax[0].axis('off')
ax[2].imshow(cells)
ax[2].set_title('Cells?')
ax[2].axis('off')
ax[1].imshow(dividing)
ax[1].set_title('Dividing nuclei?')
ax[1].axis('off')
plt.show()

#####################################################################
# Count dividing nuclei
# =====================
# Clearly, not all connected regions in the middle plot are dividing nuclei.
# On one hand, the second threshold (value of ``thresholds[1]``) appears to be
# too low to separate those very bright areas corresponding to cell division
# from relatively bright pixels otherwise present in many cells. On the other
# hand, we want a smoother image, not only to get rid of small spurious
# objects, but also to merge pairs of neighbouring objects which would
# represent two emerging nuclei in one mother cell. In a way, the segmentation
# challenge we are facing with dividing nuclei is the opposite of that with
# (touching) cells.

#####################################################################
# To find suitable values for thresholds and filtering parameters, we proceed
# by dichotomy, visually and manually.

higher_threshold = 125
dividing =  image > higher_threshold
smoother_dividing = filters.rank.mean(util.img_as_ubyte(dividing),
                                      morphology.disk(4))
binary_smoother_dividing = smoother_dividing > 20

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(binary_smoother_dividing)
ax.axis('off')
plt.show()

#####################################################################
# We are left with

print(morphology.label(binary_smoother_dividing).max())

#####################################################################
# dividing nuclei in this sample.
