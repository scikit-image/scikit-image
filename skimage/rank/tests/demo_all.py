import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from skimage import data
from skimage.morphology.selem import disk
from skimage.rank import _crank8,_crank8_percentiles
from skimage.rank import _crank16,_crank16_percentiles,_crank16_bilateral

def plot_all():
    a8 = data.camera()
    a16 = a8.astype('uint16')*16
    #    selem = np.ones((30,30),dtype='uint8')
    selem = disk(5)


    for n in dir(_crank16):
        method = eval('_crank16.%s'%n)
        t = type(method)
        if t == type(_crank8.maximum):
            print n,t
            f = method(a16,selem = selem,bitdepth=12)

            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(a16)
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(f)
            plt.colorbar()
            plt.title(method)

    for n in dir(_crank8):
        method = eval('_crank8.%s'%n)
        t = type(method)
        if t == type(_crank8.maximum):
            print n,t
            f = method(a8,selem = selem)

            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(a8)
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(f)
            plt.colorbar()
            plt.title(method)

    for n in dir(_crank8_percentiles):
        method = eval('_crank8_percentiles.%s'%n)
        t = type(method)
        if t == type(_crank8.maximum):
            print n,t
            f = method(a8,selem = selem,p0=.1,p1=.9)

            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(a8)
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(f)
            plt.colorbar()
            plt.title(method)

    for n in dir(_crank16_percentiles):
        method = eval('_crank16_percentiles.%s'%n)
        t = type(method)
        if t == type(_crank8.maximum):
            print n,t
            f = method(a16,selem = selem,bitdepth=12,p0=.1,p1=.9)

            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(a16)
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(f)
            plt.colorbar()
            plt.title(method)

    selem = disk(50)
    for n in dir(_crank16_bilateral):
        method = eval('_crank16_bilateral.%s'%n)
        t = type(method)
        if t == type(_crank8.maximum):
            print n,t
            f = method(a16,selem = selem,bitdepth=12,s0=300,s1=300)

            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(a16)
            plt.colorbar()
            plt.subplot(1,2,2)
            plt.imshow(f)
            plt.colorbar()
            plt.title(method)

            #
    plt.show()

if __name__ == '__main__':
#    plot_all()
    pprint(dir(_crank8))