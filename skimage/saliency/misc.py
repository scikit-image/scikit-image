from skimage.color import rgb2lab                                   
from scipy import ndimage                                       
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import numpy as np

def labSaliency(rgbImage,size=5,displayResult=False):
	"""
	The function calculates lab based saliency of input rgbImage

	References:
	Achanta,S. Hemami,F. Estrada,S. Susstrunk 'Frequency-tuned Salient region detection
	IEEE International Conference on Computer Vision and Pattern Recognition (CVPR 2009),
       	Inputs:
		rgbImage: input rgb image whose saliency map has to be calculated
		size: size of the gaussian filter kernel to smooth the rgbImage
	
	Outputs:
		The functions returns the saliency map of Input Image
	"""
	rgbImage = filters.gaussian_filter(rgbImage,size)
	labImage = rgb2lab(rgbImage)
	Mean = np.asarray([labImage[:,:,0].mean(),labImage[:,:,1].mean(),labImage[:,:,2].mean()])
	meanSubtracted = (labImage - Mean)**2
	saliencyMap = meanSubtracted[:,:,0] + meanSubtracted[:,:,1] + meanSubtracted[:,:,2]
	if(displayResult):
		plt.imshow(saliencyMap)
		plt.show()
	return saliencyMap
