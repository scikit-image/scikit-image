import numpy as np
from scipy.ndimage.filters import gaussian_filter as gf,maximum_filter
import itertools as itt
import math 
from math import sqrt,hypot
from numpy import arccos
from skimage.util import img_as_float


# A lot of this code is borrowed from here
# https://github.com/adonath/blob_detection/tree/master/blob_detection


def _get_local_maxima_3d(array,thresh):


    #computing max filter using all neighbors in cube
    fp = np.ones((3,3,3))
    max_array = maximum_filter(array,footprint = fp)
    peaks = (max_array == array) & (array  > thresh) 
    return np.argwhere(peaks)


def _blob_overlap(blob1,blob2):
    root2 = sqrt(2)
    
    #extent of the blob is given by sqrt(2)*scale
    r1 = blob1[2]*root2
    r2 = blob2[2]*root2
    
    
    d = hypot(blob1[0] - blob2[0],blob1[1] - blob2[1])
    
    if( d > r1 + r2):
        return 0
        
    #one blob is inside the other, the smaller blob must die
    if d < abs(r1 - r2):
        return 1
    
    area = (r1**2 * arccos((d**2 + r1**2 - r2**2)/(2 * d * r1)) 
                                + r2**2 * arccos((d**2 + r2**2 - r1**2)/(2 * d * r2)) 
                                - 0.5 * sqrt(abs((-d + r2 + r1)*(d + r2 - r1) * 
                                                 (d - r2 + r1) * (d + r2 + r2))))
    
    return area/(math.pi*(min(r1,r2)**2))
    
def _prune_blobs(array,overlap):
    
    #iterating again might eliminate more blobs, but one iteration suffices
    # for most cases
    for blob1,blob2 in itt.combinations(array,2) :
        if _blob_overlap(blob1,blob2) > overlap :
            if blob1[2] > blob2[2] :
                blob2[2] = -1 
            else:
                blob1[2] = -1
                
    return np.array([ a for a in array if a[2] > 0 ])
    

def get_blobs(image,min_sigma=1,max_sigma=20,num_sigma=50,thresh=0.25,
            overlap=.5,mode='dog'):
    
    """Finds blobs in the given grayscale image
    
    For each blob found, it's coordinates and area are returned

    Parameters
    ----------
    image : ndarray
        Input grayscale image, blobs are assumed to be light on dark
        background ( white on black )
    min_sigma : float, optional
        The minimum standard deviation for Gaussian Kernel. Keep this low to 
        detect smaller blobs
    max_sigma : float, optional
        The maximum standard deviation for Gaussian Kernel. Keep this high to
        detect larger blobs
    num_sigma : float, optional
        The number of times Gaussian Kernels are computed i.e , the length 
        of third dimension of the sace space
    thresh : float, optional
        The lower bound for scale space maxima. Local maxima smaller than thresh
        are ignored
    overlap : float, optional
        A value between 0 and 1. If the area of two blobs overlaps by a fraction
        greather than 'thresh', the smaller blob is eliminated
    mode : {'dog'}, optional
        The algorithm to use
        dog - Difference of Gaussian

    Returns
    -------
    A : ndarray
        A 2d array in which each row contains 3 values, the Y-Coordinate , the 
        X-Coordinate and the estimated area of the blob respectively


    Examples
    --------
    >>> from skimage import data,feature
    >>> blobs = feature.get_blobs(data.coins())
    >>> for blob in blobs:
    ...     print "Blob found at (%d,%d) of area %d" % (blob[1],blob[0],blob[2])
    ... 
    Blob found at (336,46) of area 2513
    Blob found at (156,53) of area 2035
    Blob found at (217,53) of area 1608
    Blob found at (276,54) of area 1231
    .
    .
    .

    """

    if(len(image.shape) != 2):
        raise ValueError("'image' must be a grayscale ")
        
    scales = np.linspace(min_sigma,max_sigma,num_sigma)
    image = img_as_float(image)
    
    if mode == 'dog':
        ds = 0.1
        #ordered from inside to out
        # compute difference of gaussian , normalize with scale space,
        # iterate over all scales, and finally put all images obtained in
        # a 3d array with np.dstack
        image_cube = np.dstack( [(gf(image,s) - gf(image,s+ds))*s**2\
                     for s in scales] )
        
    else:
        raise ValueError("Specified 'mode' is unknown")
        
    
    local_maxima = _get_local_maxima_3d(image_cube,thresh)
    #print local_maxima
    local_maxima[:,2] = scales[local_maxima[:,2]]
    #print local_maxima
    ret_val = _prune_blobs(local_maxima,overlap)
    if len(ret_val) > 0:
        ret_val[:,2] = math.pi *((ret_val[:,2]*math.sqrt(2))**2).astype(int)
        return ret_val
    else:
        return []
    
        
