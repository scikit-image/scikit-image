"""
===============================================
Identify human cells and estimate mitotic index
===============================================

In this example, we analyze a microscopy image of human cells. We use data
provided by Jason Moffat [1]_ through [CellProfiler](https://cellprofiler.org/examples/#human-cells).

.. [1] Moffat J, Grueneberg DA, Yang X, Kim SY, Kloepfer AM, Hinkle G, Piqani B, Eisenhaure TM, Luo B, Grenier JK, Carpenter AE, Foo SY, Stewart SA, Stockwell BR, Hacohen N, Hahn WC, Lander ES, Sabatini DM, Root DE (2006) "A lentiviral RNAi library for human and mouse genes applied to an arrayed viral high-content screen" Cell, 124(6):1283-98. DOI: [10.1016/j.cell.2006.01.040](https://doi.org/10.1016/j.cell.2006.01.040). PMID: 16564017

"""

import matplotlib.pyplot as plt

from skimage import io


image = io.imread('https://github.com/CellProfiler/examples/blob/master/ExampleHuman/images/AS_09125_050116030001_D03f00d0.tif?raw=true')

#####################################################################
# This image is a TIFF file. If you run into issues loading it, please
# consider using ``external.tifffile.imread`` or following:
# https://github.com/scikit-image/scikit-image/issues/4326#issuecomment-559595147

fig, ax = plt.subplots()
ax.imshow(image, cmap='gray')
plt.show()
