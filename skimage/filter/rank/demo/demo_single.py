import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology.selem import disk
import skimage.filter.rank as rank

from skimage.filter import denoise_bilateral

if __name__ == '__main__':
    a8 = data.camera()
    a16 = data.camera().astype(np.uint16)*4

    p8 = data.page()

    selem = disk(20)

    otsu = rank.otsu(p8,selem)


    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(p8)
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(otsu)
    plt.colorbar()
    plt.show()




