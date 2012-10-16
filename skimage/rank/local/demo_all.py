import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

from skimage import data
from skimage.morphology.selem import disk
import skimage.rank as rank

def plot_all():
    a8 = data.camera()
    a16 = a8.astype('uint16')*16
    selem = disk(5)

    name_list = sorted([n for n in dir(rank) if n[0] is  not '_'])
    print name_list

    for n in name_list:
        if n.rfind('bilateral')==0:
            print n
            method = eval('rank.%s'%n)
            if type(method) == type(rank.maximum):
                print method
                f8 = method(a8,selem = selem,s0=10,s1=10)
                f16 = method(a16,selem = selem,s0=10,s1=10)
                plt.figure()
                plt.subplot(2,2,1)
                plt.imshow(a8)
                plt.colorbar()
                plt.subplot(2,2,2)
                plt.imshow(f8)
                plt.colorbar()
                plt.subplot(2,2,3)
                plt.imshow(f16)
                plt.colorbar()
                plt.title(method)
    for n in name_list:
        if n.rfind('percentile')==0:
            print n
            method = eval('rank.%s'%n)
            if type(method) == type(rank.maximum):
                print method
                f8 = method(a8,selem = selem,p0=.1,p1=.9)
                f16 = method(a16,selem = selem,p0=.1,p1=.9)
                plt.figure()
                plt.subplot(2,2,1)
                plt.imshow(a8)
                plt.colorbar()
                plt.subplot(2,2,2)
                plt.imshow(f8)
                plt.colorbar()
                plt.subplot(2,2,3)
                plt.imshow(f16)
                plt.colorbar()
                plt.title(method)
    for n in name_list:
        if n.find('percentile')==-1 and n.find('bilateral')==-1:
            print n
            method = eval('rank.%s'%n)
            if type(method) == type(rank.maximum):
                print method
                f8 = method(a8,selem = selem)
                f16 = method(a16,selem = selem)
                plt.figure()
                plt.subplot(2,2,1)
                plt.imshow(a8)
                plt.colorbar()
                plt.subplot(2,2,2)
                plt.imshow(f8)
                plt.colorbar()
                plt.subplot(2,2,3)
                plt.imshow(f16)
                plt.colorbar()
                plt.title(method)
    plt.show()

if __name__ == '__main__':
    plot_all()
    pprint(dir(rank))