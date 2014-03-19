"""author:tabish
"""
from skimage.color import rgb2lab                                   
from scipy.ndimage import filters
import matplotlib.pyplot as plt
import numpy as np

def lab_saliency(rgb_image, size=5, display_result=False):
    """The function calculates lab color space based(Frequency-tuned) saliency
    of input rgb_image

    References:
        Achanta,S. Hemami,F. Estrada,S. Susstrunk 'Frequency-tuned Salient region detection
        IEEE International Conference on Computer Vision and Pattern Recognition (CVPR 2009),
    Inputs:
        rgb_image: input rgb image whose saliency map has to be calculated
        size: size of the gaussian filter kernel to smooth the rgb_image

    Outputs:
        The functions returns the saliency map of Input Image
    """
    #smooth the input rgb_image
    rgb_image = filters.gaussian_filter(rgb_image, size)
    #convert to lab color space
    lab_image = rgb2lab(rgb_image)
    mean = np.asarray([lab_image[:, :, 0].mean(), lab_image[:, :, 1].mean(), lab_image[:, :, 2].mean()])
    mean_subtracted = (lab_image - mean)**2
    saliency_map = mean_subtracted[:, :, 0] + mean_subtracted[:, :, 1] + mean_subtracted[:, :, 2]
    if(display_result):
        plt.imshow(saliency_map)
        plt.show()
    return saliency_map
