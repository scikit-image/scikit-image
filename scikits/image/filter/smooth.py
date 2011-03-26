"""smooth.py - smoothing of images

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky

"""
import numpy as np
import scipy.linalg

def smooth_with_noise(image, bits):
    """Smooth the image with a per-pixel random multiplier
    
    image - the image to perturb
    bits - the noise is this many bits below the pixel value
    
    The noise is random with normal distribution, so the individual pixels
    get either multiplied or divided by a normally distributed # of bits
    """
    
    rr = np.random.RandomState()
    rr.seed(0)
    r = rr.normal(size=image.shape)
    delta = pow(2.0,-bits)
    image_copy = np.clip(image, delta, 1)
    result = np.exp2(np.log2(image_copy + delta) * r + 
                     (1-r) * np.log2(image_copy))
    result[result>1] = 1
    result[result<0] = 0
    return result

def smooth_with_function_and_mask(image, function, mask):
    """Smooth an image with a linear function, ignoring the contribution of masked pixels
    
    image - image to smooth
    function - a function that takes an image and returns a smoothed image
    mask  - mask with 1's for significant pixels, 0 for masked pixels
    
    This function calculates the fractional contribution of masked pixels
    by applying the function to the mask (which gets you the fraction of
    the pixel data that's due to significant points). We then mask the image
    and apply the function. The resulting values will be lower by the bleed-over
    fraction, so you can recalibrate by dividing by the function on the mask
    to recover the effect of smoothing from just the significant pixels.
    """
    not_mask               = np.logical_not(mask)
    bleed_over             = function(mask.astype(float))
    masked_image           = np.zeros(image.shape, image.dtype)
    masked_image[mask]     = image[mask]
    smoothed_image         = function(masked_image)
    output_image           = smoothed_image / (bleed_over + np.finfo(float).eps)
    return output_image

def circular_gaussian_kernel(sd,radius):
    """Create a 2-d Gaussian convolution kernel
    
    sd     - standard deviation of the gaussian in pixels
    radius - build a circular kernel that convolves all points in the circle
             bounded by this radius 
    """
    i,j = np.mgrid[-radius:radius+1,-radius:radius+1].astype(float) / radius
    mask = i**2 + j**2 <= 1
    i = i * radius / sd
    j = j * radius / sd

    kernel = np.zeros((2*radius+1,2*radius+1))
    kernel[mask] = np.e ** (-(i[mask]**2+j[mask]**2) /
                            (2 * sd **2))
    #
    # Normalize the kernel so that there is no net effect on a uniform image
    #
    kernel = kernel / np.sum(kernel)
    return kernel

def fit_polynomial(pixel_data, mask):
    '''Return an "image" which is a polynomial fit to the pixel data
    
    Fit the image to the polynomial Ax**2+By**2+Cxy+Dx+Ey+F
    '''
    mask = np.logical_and(mask,pixel_data > 0)
    if not np.any(mask):
        return pixel_data
    x,y = np.mgrid[0:pixel_data.shape[0],0:pixel_data.shape[1]]
    x2 = x*x
    y2 = y*y
    xy = x*y
    o  = np.ones(pixel_data.shape)
    a = np.array([x[mask],y[mask],x2[mask],y2[mask],xy[mask],o[mask]])
    coeffs = scipy.linalg.lstsq(a.transpose(),pixel_data[mask])[0]
    output_pixels = np.sum([coeff * index for coeff, index in
                            zip(coeffs, [x,y,x2,y2,xy,o])],0)
    output_pixels[output_pixels > 1] = 1
    output_pixels[output_pixels < 0] = 0
    return output_pixels
