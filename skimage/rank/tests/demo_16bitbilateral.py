import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology import disk
import skimage.rank as rank

if __name__ == '__main__':
    a8 = (data.coins()).astype('uint8')

    a16 = (data.coins()).astype('uint16')*16
    selem = np.ones((20,20),dtype='uint8')
    f1 = rank.percentile_mean(a8,selem = selem,p0=.1,p1=.9)
    f2 = rank.bilateral_mean(a16,selem = selem,s0=500,s1=500)

    selem = disk(50)
    f3 = rank.equalize(a16,selem = selem)

    plt.figure()
    plt.imshow(np.hstack((a8,f1)))
    plt.colorbar()

    plt.figure()
    plt.imshow(np.hstack((a16,f2)))
    plt.colorbar()

    plt.figure()
    plt.imshow(np.hstack((a16,f3)))
    plt.colorbar()

    plt.show()
