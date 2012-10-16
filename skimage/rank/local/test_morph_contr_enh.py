import numpy as np
import matplotlib.pyplot as plt
import gdal

from skimage.morphology import disk
import skimage.rank as rank

filename = 'iko_pan_Ja1.tif'
im16 = gdal.Open(filename).ReadAsArray().astype(np.uint16)

plt.figure()
plt.imshow(im16,cmap=plt.cm.gray)
plt.colorbar()

f0 = rank.median(im16,disk(1))
f1 = rank.bilateral_mean(im16,disk(20),s0=200,s1=200)
f2 = rank.equalize(f1,disk(10))
f3 = rank.bottomhat(f1,disk(1))

plt.figure()
plt.imshow(f2,cmap=plt.cm.gray,interpolation='nearest')
plt.colorbar()

plt.show()




