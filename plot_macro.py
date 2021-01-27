import scipy.io as sio
import numpy as np
import seaborn as sns
import os, glob, sys
import pandas as pd
from matplotlib import pyplot as plt
from skimage import segmentation, filters
from collections import defaultdict
from scipy import ndimage as ndi

from timeit import default_timer as timer

def impute_blanks(data):
    #skip 6,6 outside border
    for j in range(7, 121):  #y axis is first
        for i in range(7, 121):
            if np.linalg.norm(data[j][i]) == 0.0:
                #try: shortened range means we don't need to try-except
                vertical = np.dot(data[j+1][i], data[j-1][i])
                horizontal = np.dot(data[j][i-1], data[j][i+1])
                if horizontal > 0.001 and vertical > 0.001: 
                    data[j][i] = (data[j+1][i]+data[j-1][i]+data[j][i+1]+data[j][i-1])/4
    return data

def smooth_data(data):
    #a few different filters to try - median, gaussian and 3x3 boxcar rolling average
    #can try to use ndimage instead of filters for gaussian etc

    smoothed = ndi.uniform_filter(data,size=3,mode='constant')
    #smoothed = filters.median(data)
    #smoothed = filters.gaussian(data,preserve_range=True)#,truncate=3.0)

    return smoothed

def get_cube(file):
    data = sio.loadmat(file)
    cube = np.array(data['cube']['betterRefl'][0][0])
    cube[np.isnan(cube)] = 0
    wn = np.array(data['cube']['wn'][0][0])
    return cube, wn

def preprocess(cube):
    #load data, set NaNs to 0

    #fill in blank pixels
    time1 = timer()
    filled_in_data = impute_blanks(cube)
    time2 = timer()
   # timeit.timeit('impute_blanks(cube)', 'from __main__ import impute_blanks, cube', number=1))
    #boxcar smoothing
    smoothed_data = smooth_data(filled_in_data)
    time3 = timer()

    print('impute: ', time2 - time1)
    print('smooth: ', time3 - time2)
    return smoothed_data

def get_segments(data):
    #do segmentation here
    time1 = timer()
    labels = segmentation.slic(data, n_segments=20, compactness=1, convert2lab=False, 
        slic_zero=False, start_label=1, max_iter=35)
    time2 = timer()
    segmentation.slic(data, n_segments=20, compactness=1, convert2lab=False, 
        slic_zero=False, start_label=1, max_iter=25)
    time3 = timer()
    labels = segmentation.slic(data, n_segments=20, compactness=1, convert2lab=False, 
        slic_zero=True, start_label=1, max_iter=35)
    time4 = timer()

    print('SLIC Algs: ', time2 -time1, time3-time2, time4-time3)
    
    
    return labels

def get_spectra(segment_labels, cube, wn):
    #get avg spectra for each segment
    segment_spectra = defaultdict(list)
    num_labels = np.max(segment_labels) + 1 #python offset
    for y, col in enumerate(segment_labels):
        for x, row in enumerate(col):
            try:
                segment_spectra[segment_labels[y][x]].append(cube[y][x])
            except:
                segment_spectra[segment_labels[y][x]] = cube[y][x]

    spectra = defaultdict(list)
    for cluster in range(1,num_labels):
        spectra[cluster] = np.mean(segment_spectra[cluster], axis=0)

    multispectra = pd.DataFrame.from_dict(spectra)
    multispectra['wn'] = wn #[0]
    multispectra.set_index('wn')

    return multispectra

def get_axis(cube):
    maxes = np.max(cube,axis=2)
    lower = np.percentile(maxes,10)
    upper = np.percentile(maxes,90)
    return lower,upper

def create_plots(image_data, segment_data, spectra_data, lower, upper):
    #create 3 plots, save out output

    #looking at median value for best 'image'
    image = np.median(image_data, axis=2)

    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(60, 20))#,sharey=True)

    sns.set(font_scale=4)
    ax1.tick_params(labelsize='small')
    sns.heatmap(image,vmin=lower,vmax=upper, ax=ax1)
    num_seg = len(np.unique(segment_data))
    cmap = sns.color_palette("hls",num_seg)
    ax2.tick_params(labelsize='small')
    sns.heatmap(segment_data, cmap=cmap, ax=ax2)
    ax3.tick_params(labelsize='medium')
    keys = list(spectra_data.columns)
    for key in keys[:-1]: #remove 'wn' at end
        #sns.lineplot(x='wn',y=key,data=spectra_data,label=key,ax=ax3, linewidth=3, hue_order=keys[:-1])
        ax3.plot(spectra_data['wn'], spectra_data[key], label=key)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.savefig("multiplot_test2.png")

if __name__ == "__main__":
    #go to data, and get data
    os.chdir('/home/block-am/Documents/SLIC Test Data/')
    #cube, wn = get_cube('SetD_sample16_03-HP-02') #for debugging purposes
    #cube, wn = get_cube('SetD_sample16_01-HP-02') #for debugging purposes
    cube, wn = get_cube('SetD_sample18_02-HP-02') #for debugging purposes
    #cube, wn = get_cube('paper_grid') #for debugging purposes
    #	cube, wn = get_cube(sys.argv[1])
    lower,upper = get_axis(cube)
    img_data = preprocess(cube)
    segment_data = get_segments(img_data)
    spectra_data = get_spectra(segment_data,cube,wn)
    create_plots(img_data,segment_data,spectra_data,lower,upper)
