from skimage import img_as_float                                                       
from skimage.color import rgb2gray                                  
from scipy import fftpack, ndimage                                       
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import numpy as np

def spectralResidue(rgbImage,sigma = 3,displayResult = False):
	"""
	This Function computes Spectral Residue Based Saliency map

	References:
	Xiaodi Hou,Liquin Zhang, Saliency Detection: A spectral residue approach
	CVPR 2007

	Inputs
	rgbimage: rgb input image for computation of saliency
	sigma: 	  standard deviation of the smoothing filter for
		      smoothed log magnitude of input rgbimage
	displayResult: set 1 if you want the result to be 
		           displayed
	Outputs:
	function return the saliency map of the rgbimage.

	Examples:
	>>filename = 'image.jpg'
	>>rgbImage = imread(filename);
	>>spectralResidue(rgbImage,displayResult = 1)
	"""
#handle cases where the input isn't an rgbimage
	try:
		grayImage = rgb2gray(rgbImage)
	except:
		grayImage = rgbImage
	grayImage = img_as_float(grayImage)
# Spectral Residue Computation
  	#compute fourier transform on image
	fftGrayImage = fftpack.fft2(grayImage)                                                   
	logMagnitude = np.log(np.abs(fftGrayImage))                                          
	phase = np.angle(fftGrayImage)     
	
	#smooth the log magnitude response  
	avgLogMagnitude = filters.gaussian_filter(logMagnitude,sigma,  mode="nearest")            
	
	#compute spectral residue
	spectralResidual = logMagnitude - avgLogMagnitude                                 
	#find inverse fourier transform of spectral residue to obtain the saliency map
	saliencyMap = np.abs(fftpack.ifft2(np.exp(spectralResidual + 1j * phase))) ** 2
	saliencyMap = ndimage.gaussian_filter(saliencyMap, sigma=3)
	#display option
	if(displayResult):
		plt.imshow(saliencyMap)
		plt.show()
	return saliencyMap
