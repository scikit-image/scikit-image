"""
=====================
Local Autolevel
=====================

Local autolevel stretch local histogram between 0 and max_graylevel (e.g. 255 for 8 bit image).
The following code shows the difference between autolevel and percentile auto_level where [min,max] interval
is replaced by [p0,p1] percentiles interval

"""
import matplotlib.pyplot as plt
import numpy as np

from skimage import data

from skimage.filter.rank import percentile_autolevel,autolevel
from skimage.morphology import disk


image = data.camera()

selem = disk(20)
loc_autolevel = autolevel(image,selem=selem)
loc_perc_autolevel0 = percentile_autolevel(image,selem=selem,p0=.00,p1=1.0)
loc_perc_autolevel1 = percentile_autolevel(image,selem=selem,p0=.01,p1=.99)
loc_perc_autolevel2 = percentile_autolevel(image,selem=selem,p0=.05,p1=.95)
loc_perc_autolevel3 = percentile_autolevel(image,selem=selem,p0=.1,p1=.9)

loc_perc_autolevel = np.hstack((loc_perc_autolevel0,loc_perc_autolevel1,loc_perc_autolevel2,loc_perc_autolevel3))

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2 = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Image')

ax1.imshow(loc_autolevel)
ax1.set_title('Autolevel')

ax2.imshow(loc_perc_autolevel,vmin=0,vmax=255)
ax2.set_title('percentile autolevel 0%,1%,5% and 10%')

for ax in axes:
    ax.axis('off')

plt.show()
