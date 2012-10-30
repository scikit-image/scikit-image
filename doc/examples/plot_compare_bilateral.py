"""
====================================================
Bilateral comparison
====================================================

In this example, we compare both bilateral implementation

* filter.denoise_bilateral
* filter.rank.bilateral_mean

The first filter implements a spatial-gaussian and spectral-gaussian kernel bilateral filter whereas the latter implements
a cylindrical kernel bilateral filter i.e. spatial-flat and spectral-flat kernel.

The timing comparison is just for information since the kernel are not the same.

"""

import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.filter._denoise import denoise_bilateral
from skimage.filter.rank import bilateral_mean
from skimage.morphology import disk
from skimage.filter import denoise_bilateral
import time

def exec_and_timeit(func):
    """ Decorator that returns both function results and execution time
    (result, ms)
    """
    def wrapper(*arg):
        t1 = time.time()
        res = func(*arg)
        t2 = time.time()
        ms = (t2-t1)*1000.0
        return (res,ms)
    return wrapper


@exec_and_timeit
def den_bil(image):
    return denoise_bilateral(a8,win_size=20,sigma_range=255,sigma_spatial=1)[:,:,0]*255

@exec_and_timeit
def rank_bil(image):
    return bilateral_mean(a8.astype(np.uint16),disk(20),s0=10,s1=10)

a8 = data.camera()
selem = disk(10)

f1,t1 = den_bil(a8)
f2,t2 = rank_bil(a8)

# display results
fig, axes = plt.subplots(nrows=2, figsize=(15,10))
ax0, ax1= axes

ax0.imshow(np.hstack((f1,a8-f1)))
ax0.set_title('denoise bilateral (%f ms)'%t1)
ax1.imshow(np.hstack((f2,a8-f1)))
ax1.set_title('bilateral mean (%f ms)'%t2)
plt.show()
