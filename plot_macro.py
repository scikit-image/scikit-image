import scipy.io as sio
import numpy as np
import seaborn as sns
import os, glob, sys
import pandas as pd
from matplotlib import pyplot as plt
from skimage import segmentation
from collections import defaultdict

def impute_blanks(data):
	#skip 6,6 outside border
	#new_data = np.zeros([128,128])
	for j in range(7,121):
		for i in range(7,121):
			if data[j][i] == 0.0:
				#try: shortened range means we don't need to try-except
				vertical = data[j+1][i] + data[j-1][i]
				horizontal = data[j][i-1] + data[j][i+1]
				if horizontal > 0.001 and vertical > 0.001: 
					data[j][i] = (vertical+horizontal)/4
	return data

def get_cube(file):
	data = sio.loadmat(file)
	cube = np.array(data['cube']['betterRefl'][0][0])
	cube[np.isnan(cube)] = 0
	wn = np.array(data['cube']['wn'][0][0])
	return cube, wn

def preprocess(cube):
	#load data, set NaNs to 0

	#handle broken pixels
	avg_data = np.median(cube,axis=2)
	filled_in_data = impute_blanks(avg_data)	

	#wn = data['cube']['wn'][0][0] #maybe one less [0]?
	#average results (boxcar)
	
	return filled_in_data#cube

def get_segments(data):
	#do segmentation here
	labels = segmentation.slic(data, n_segments=20, compactness=5, multichannel=True, convert2lab=False)
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
	for cluster in range(num_labels):
		spectra[cluster] = np.mean(segment_spectra[cluster],axis=0)

	multispectra = pd.DataFrame.from_dict(spectra)
	multispectra['wn'] = wn #[0]
	multispectra.set_index('wn')

	return multispectra

def get_axis(cube):
	maxes = np.max(cube,axis=2)
	lower = np.percentile(maxes,10)
	upper = np.percentile(maxes,90)
	return lower,upper

def create_plots(image_data,segment_data,spectra_data,lower,upper):
    #create 3 plots, save out output

	fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(60,20))#,sharey=True)

	sns.set(font_scale=4)
	ax1.tick_params(labelsize='small')
	sns.heatmap(image_data,vmin=lower,vmax=upper, ax=ax1)
	num_seg = len(np.unique(segment_data))
	cmap = sns.color_palette("hls",num_seg)
	ax2.tick_params(labelsize='small')
	sns.heatmap(segment_data, cmap=cmap, ax=ax2)
	ax3.tick_params(labelsize='medium')
	keys = list(spectra_data.columns)
	#keys.remove('wn')
	for key in keys[:-1]:
		sns.lineplot(x='wn',y=key,data=spectra_data,label=key,ax=ax3, linewidth=3)

	plt.savefig("multiplot_test.png")
	#plt.clf()

if __name__ == "__main__":
	#go to data, and get data
	os.chdir('/home/block-am/Documents/SLIC Test Data/')
	cube, wn = get_cube('paper_grid') #for debugging purposes
	#	cube, wn = get_cube(sys.argv[1])
	lower,upper = get_axis(cube)
	img_data = preprocess(cube)
	segment_data = get_segments(img_data)
	spectra_data = get_spectra(segment_data,cube,wn)
	create_plots(img_data,segment_data,spectra_data,lower,upper)