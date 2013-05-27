"""
====================
Percentile filters
====================
Replaces the pixel by the local gray level (inside a given structuring element) such that this gray level
is higher or equal to percentile p0

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology import disk
from skimage.filter import rank
from skimage import color

lena = np.array(256 * color.rgb2gray(data.lena()), dtype=np.uint8)
selem = disk(5)

# display results
fig, axes = plt.subplots(ncols=4, nrows=4, figsize=(15, 10))

max_row = 4
max_col = 4
p0 = np.linspace(0, 1, max_col * max_row, endpoint=True)
i = 0
for row in axes:
    for ax in row:
        perc = rank.percentile(lena, selem=selem, p0=p0[i])
        ax.imshow(perc, cmap=plt.cm.gray)
        ax.set_title('perc. : %.2f' % p0[i])
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        i += 1

plt.show()
