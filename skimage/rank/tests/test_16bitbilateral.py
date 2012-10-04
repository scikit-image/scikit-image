import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.rank import crank8_percentiles,crank16_bilateral

if __name__ == '__main__':
    a8 = (data.coins()).astype('uint8')

    a16 = (data.coins()).astype('uint16')*16
    selem = np.ones((20,20),dtype='uint8')
    f1 = crank8_percentiles.mean(a8,selem = selem,p0=.1,p1=.9)
    f2 = crank16_bilateral.mean(a16,selem = selem,bitdepth=12,s0=500,s1=500)

    plt.figure()
    plt.imshow(np.hstack((a8,f1)))
    plt.colorbar()

    plt.figure()
    plt.imshow(np.hstack((a16,f2)))
    plt.colorbar()

    plt.show()
