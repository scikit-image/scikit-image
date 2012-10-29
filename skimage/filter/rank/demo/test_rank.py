import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology.selem import disk
import skimage.filter.rank as rank

print dir(rank)

print rank.mean
print rank.percentile_mean
print rank.bilateral_mean

a8 = data.camera()
a16 = a8.astype('uint16')*16
selem = disk(10)

f8 = rank.mean(a8,selem)
f16 = rank.mean(a16,selem)

plt.figure()
plt.imshow(np.hstack((a8,f8)))
plt.colorbar()
plt.figure()
plt.imshow(np.hstack((a16,f16)))
plt.colorbar()

f8 = rank.percentile_mean(a8,selem,p0=.1,p1=.9)
f16 = rank.percentile_mean(a16,selem,p0=.1,p1=.9)

plt.figure()
plt.imshow(np.hstack((a8,f8)))
plt.colorbar()
plt.figure()
plt.imshow(np.hstack((a16,f16)))
plt.colorbar()

plt.show()




