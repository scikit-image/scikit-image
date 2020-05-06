"""
===============================================
Identify human cells and estimate mitotic index
===============================================

In this example, we analyze a microscopy image of human cells. We use data
provided by Jason Moffat [1]_ through [CellProfiler](https://cellprofiler.org/examples/#human-cells).

.. [1] Moffat J, Grueneberg DA, Yang X, Kim SY, Kloepfer AM, Hinkle G, Piqani B, Eisenhaure TM, Luo B, Grenier JK, Carpenter AE, Foo SY, Stewart SA, Stockwell BR, Hacohen N, Hahn WC, Lander ES, Sabatini DM, Root DE (2006) "A lentiviral RNAi library for human and mouse genes applied to an arrayed viral high-content screen" Cell, 124(6):1283-98. DOI: [10.1016/j.cell.2006.01.040](https://doi.org/10.1016/j.cell.2006.01.040). PMID: 16564017

"""

import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from skimage.filters import threshold_multiotsu


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

# Thresholding
# ============
# We are thus interested in two thresholds: one separating the cells from the
# background, the other separating the dividing nuclei (brighter spots) from
# the cytoplasm of their respective mother cells (and, intensity-wise, from
# the other cells). To separate these three different classes of pixels, we
# resort to :ref:`sphx-glr-auto-examples-segmentation-plot-multiotsu-py`.

thresholds = threshold_multiotsu(image)
regions = np.digitize(image, bins=thresholds)

fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].imshow(image)
ax[0].set_title('Original')
ax[0].axis('off')
ax[1].imshow(regions)
ax[1].set_title('Multi-Otsu thresholding')
ax[1].axis('off')
plt.show()
