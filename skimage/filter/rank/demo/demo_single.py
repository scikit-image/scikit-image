import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology.selem import disk
import skimage.filter.rank as rank

from skimage.filter import denoise_bilateral

if __name__ == '__main__':
    a8 = data.camera()
    a16 = data.camera().astype(np.uint16)*4
    selem = disk(10)

    f8= rank.percentile_autolevel(a8,selem,p0=.0,p1=1.)
    f16= rank.autolevel(a16,selem)
    f16p= rank.percentile_autolevel(a16,selem,p0=.0,p1=1.)

    den = denoise_bilateral(a8,win_size=10,sigma_range=10,sigma_spatial=2)[:,:,0]
    f16b= rank.bilateral_mean(a8.astype(np.uint16),disk(10),s0=10,s1=10)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(den)
    plt.subplot(1,2,2)
    plt.imshow(f16b)
    plt.show()

    print f16==f16p

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(f16)
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(f16p)
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(f16p-f16)
    plt.colorbar()
    plt.show()

    print f16
    print f16p



