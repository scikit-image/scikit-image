import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from skimage import data
from skimage.morphology.selem import disk
import skimage.rank as rank


if __name__ == '__main__':
    a8 = data.camera()
    a16 = data.camera().astype(np.uint16)
    selem = disk(10)

    f8= rank.mean(a8,selem)
    f16= rank.mean(a16,selem)

    print f8==f16

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(a8)
    plt.subplot(1,2,2)
    plt.imshow(f8-f16)
    plt.show()




