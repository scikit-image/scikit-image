from skimage import img_as_float                                                       
from skimage.color import rgb2gray                                  
from scipy import fftpack, ndimage                                       
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import numpy as np

def sr_saliency(rgb_image, sigma = 3, display_result = False):
    """This Function computes Spectral Residue Based Saliency map


    References:
        Xiaodi Hou,Liquin Zhang, Saliency Detection: A spectral residue approach
        CVPR 2007

    Inputs
        rgbimage:  rgb input image for computation of saliency
        sigma(optional):  standard deviation of the smoothing filter for
	   	  smoothed log magnitude of input rgbimage
        display_result(optional):  set 1 if you want the result to be 
	         		  displayed
    Outputs:
        function return the saliency map of the rgbimage.

    Examples:
        >>filename = 'image.jpg'
        >>rgb_image = imread(filename);
        >>sr_saliency(rgb_image,display_result = 1)
    """
    #handle cases where the input isn't an rgbimage
    try:
        gray_image = rgb2gray(rgb_image)
    except:
        gray_image = rgb_image
    gray_image = img_as_float(gray_image)
    # Spectral Residue Computation
    
    #compute fourier transform on image
    fft_gray_image = fftpack.fft2(gray_image)                                                   
    log_magnitude = np.log(np.abs(fft_gray_image))                                          
    phase = np.angle(fft_gray_image)     

    #smooth the log magnitude response  
    avg_log_magnitude = filters.gaussian_filter(log_magnitude, sigma, mode="nearest")            

    #compute spectral residue
    spectral_residual = log_magnitude - avg_log_magnitude                                 
    
    #find inverse fourier transform of spectral residue to obtain the saliency map
    saliency_map = np.abs(fftpack.ifft2(np.exp(spectral_residual + 1j * phase))) ** 2
    saliency_map = ndimage.gaussian_filter(saliency_map, sigma=3)
    
    #display option
    if(display_result):
        plt.imshow(saliency_map)
        plt.show()
    return saliency_map
