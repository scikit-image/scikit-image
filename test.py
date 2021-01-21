#!/bin/python

## test slic algs

import scipy.io as sio
import numpy as np
import seaborn as sns
import os, glob

from skimage.future import graph
from skimage import segmentation, color, filters, io, data
from matplotlib import pyplot as plt

def get_files():        

    path = '/home/block-am/Documents/SLIC Test Data/'
    os.chdir(path)
    files = glob.glob('*.mat') #os.listdir(path)

    filenames = []
    samples = []

    for file in files:
        filename = os.path.join(path, file[:-4])
        filenames.append(filename)
        if file[:3] == 'Set':
            samples.append(file[11:-4])
        else:
            samples.append('paper_grid')

    return filenames, samples

def create_plots(files,samples):

    for i,file in enumerate(files):
        print(i,samples[i])

        #load data, remove NaNs
        data_tictac = sio.loadmat(file)
        cube = np.array(data_tictac['cube']['betterRefl'][0][0])
        cube[np.isnan(cube)] = 0

        #plot 'pretty' data
        plot_data = np.median(cube,axis=2)
        maxes = np.max(cube,axis=2)
        lower = np.percentile(maxes, 10)
        upper = np.percentile(maxes, 90)

        #print(lower, upper)
        ax = sns.heatmap(plot_data,vmin=lower,vmax=upper)
        plt.savefig("output-"+samples[i]+".png")
        plt.clf()
        #plt.show()

if __name__ == "__main__":
    
    os.chdir('/home/block-am/Documents/SLIC Test Data/')
    files, samples = get_files()
    create_plots(files,samples)