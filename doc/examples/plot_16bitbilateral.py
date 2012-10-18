"""
==============================
Simplified bilateral filtering
==============================

to complete

"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology import disk
import skimage.filter.rank as rank

a8 = (data.coins()).astype('uint8')

a16 = (data.coins()).astype('uint16')*16
selem = np.ones((20,20),dtype='uint8')
f1 = rank.percentile_mean(a8,selem = selem,p0=.1,p1=.9)
f2 = rank.bilateral_mean(a16,selem = selem,s0=500,s1=500)
selem = disk(50)
f3 = rank.equalize(a16,selem = selem)

# display results
fig, axes = plt.subplots(nrows=3, figsize=(15,15))
ax0, ax1, ax2 = axes

ax0.imshow(np.hstack((a8,f1)))
ax0.set_title('percentile mean')
ax1.imshow(np.hstack((a16,f2)))
ax1.set_title('bilateral mean')
ax2.imshow(np.hstack((a16,f3)))
ax2.set_title('local equalization')
plt.show()
