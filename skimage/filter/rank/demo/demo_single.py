import numpy as np
import matplotlib.pyplot as plt

from skimage import data
from skimage.morphology.selem import disk
import skimage.filter.rank as rank
from skimage.filter import threshold_otsu

if __name__ == '__main__':
    p8 = data.page()

    radius = 10
    selem = disk(radius)

    loc_otsu = rank.otsu(p8,selem)
    t_glob_otsu = threshold_otsu(p8)
    glob_otsu = p8>=t_glob_otsu


    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(p8,cmap=plt.cm.gray)
    plt.xlabel('original')
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(loc_otsu,cmap=plt.cm.gray)
    plt.xlabel('local Otsu ($radius=%d$)'%radius)
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(p8>=loc_otsu,cmap=plt.cm.gray)
    plt.xlabel('original>=local Otsu'%t_glob_otsu)
    plt.subplot(2,2,4)
    plt.imshow(glob_otsu,cmap=plt.cm.gray)
    plt.xlabel('global Otsu ($t=%d$)'%t_glob_otsu)
    plt.show()




