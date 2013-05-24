"""
====================
Percentile filters
====================


"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology import disk
import skimage.filter.rank as rank

import skimage.color as color

lena = np.array(256*color.rgb2gray(data.lena()),dtype=np.uint8)
selem = disk(5)

lena05_8bit = rank.percentile(lena,selem=selem,p0=0.05)
lena10_8bit = rank.percentile(lena,selem=selem,p0=0.10)

# display results
fig, axes = plt.subplots(ncols=4, figsize=(15, 10))
ax0, ax1, ax2, ax3 = axes

ax0.imshow(lena)
ax0.set_title('original')
ax1.imshow(lena05_8bit)
ax1.set_title('.05')
ax2.imshow(lena10_8bit)
ax2.set_title('.1')
ax3.imshow(lena10_8bit-lena05_8bit)
ax3.set_title('.1')
print rank.percentile.__doc__

plt.show()
