"""
=====================
Local Autolevel
=====================

Local autolevel stretch local histogram between 0 and max_graylevel (e.g. 255 for 8 bit image).
The following code shows the difference between autolevel and percentile auto_level where [min,max] interval
is replaced by [p0,p1] percentiles interval

"""
import matplotlib.pyplot as plt

from skimage import data

from skimage.rank import percentile_autolevel,autolevel
from skimage.morphology import disk


image = data.camera()

selem = disk(20)
loc_autolevel = autolevel(image,selem=selem)
loc_perc_autolevel = percentile_autolevel(image,selem=selem,p0=.0,p1=1.0)

assert (loc_autolevel==loc_perc_autolevel).all()

loc_perc_autolevel = percentile_autolevel(image,selem=selem,p0=.01,p1=.99)

fig, axes = plt.subplots(nrows=3, figsize=(7, 8))
ax0, ax1, ax2 = axes
plt.gray()

ax0.imshow(image)
ax0.set_title('Image')

ax1.imshow(loc_autolevel)
ax1.set_title('Autolevel')

ax2.imshow(loc_perc_autolevel,vmin=0,vmax=255)
ax2.set_title('percentile autolevel')

for ax in axes:
    ax.axis('off')

plt.show()
