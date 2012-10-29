import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology.selem import disk
import skimage.filter.rank as rank


if __name__ == '__main__':
    a8 = data.camera()
    a16 = data.camera().astype(np.uint16)*4
    selem = disk(10)

    f8= rank.percentile_autolevel(a8,selem,p0=.0,p1=1.)
    f16= rank.autolevel(a16,selem)
    f16p= rank.percentile_autolevel(a16,selem,p0=.0,p1=1.)

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



