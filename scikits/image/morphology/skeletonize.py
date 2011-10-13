"""skeletonize.py - Use an iterative thinning algorithm to find the
                    skeletons of binary objects in an image.

Original author: Neil Yager
"""

import numpy as np
from scipy.ndimage import correlate
from .. import util

def skeletonize(image):
    """
    Return a single pixel wide skeleton of all connected components 
     in a binary image. 
     
    The algorithm works by making successive passes of the image, 
     removing pixels on object borders. This continues until no
     more pixels can be removed.  The image is correlated with a
     mask that assigns each pixel a number in the range [0...255]
     corresponding to each possible pattern of its 8 neighbouring
     pixels. A look up table is then used to assign the pixels a
     value of 0, 1, 2 or 3, which are selectively removed during
     the iterations. 

    Parameters
    ----------
    
    image: ndarray (2D)
        A binary image containing the objects to be skeletonized. '1' 
         represents foreground, and '0' represents background. It 
         also accepts arrays of boolean values where True is foreground.
    
    Notes
    -----
    
    This implementation gives different results than a medial 
     axis transforrmation, which can be can be implemented using  
     morphological operations. This implementation is generally much
     faster.
    
    Returns 
    -------
    
    out: ndarray
        A matrix containing the thinned image
   
    References
    ----------
    A fast parallel algorithm for thinning digital patterns, 
    T. Y. ZHANG and C. Y. SUEN, Communications of the ACM,
    March 1984, Volume 27, Number 3

    
    Examples
    --------
    """
    
    # look up table 
    lut = [ 0,0,0,1,0,0,1,3,0,0,3,1,1,0,1,3,0,0,0,0,0,0,0,0,2,0,2,0,3,0,3,3,
            0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,3,0,2,2,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            2,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,3,0,0,0,0,0,0,0,3,0,0,0,3,0,2,0,
            0,1,3,1,0,0,1,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
            3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            2,3,1,3,0,0,1,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            2,3,0,1,0,0,0,1,0,0,0,0,0,0,0,0,3,3,0,1,0,0,0,0,2,2,0,0,2,0,0,0]

    # convert to unsigned int (this should work for boolean values)
    skeleton = np.array(image).astype(np.uint8)

    # check some properties of the input image:
    #  - 2D
    #  - binary image with only 0's and 1's 
    if skeleton.ndim != 2:
        raise ValueError('Skeletonize requires a 2D array')    
    for val in np.unique(skeleton):
        if val not in [0, 1]:
            raise ValueError('Invalid value in the image: %d'%(val))
    
    # create the mask that will assign a unique value based on the
    #  arrangement of neighbouring pixels
    mask = np.array([[  1,  2,  4],
                     [128,  0,  8],
                     [ 64, 32, 16]], np.uint8)

    pixelRemoved = True
    while pixelRemoved:
        pixelRemoved = False;

        # pass 1 - remove the 1's and 3's
        neighbours = correlate(skeleton, mask, mode='constant')
        neighbours[skeleton == 0] = 0
        codes = np.take(lut, neighbours)
        if np.any(codes == 1): 
            pixelRemoved = True
            skeleton[codes == 1] = 0
        if np.any(codes == 3): 
            pixelRemoved = True
            skeleton[codes == 3] = 0

        # pass 2 - remove the 2's and 3's
        neighbours = correlate(skeleton, mask, mode='constant')
        neighbours[skeleton == 0] = 0
        codes = np.take(lut, neighbours)
        if np.any(codes == 2): 
            pixelRemoved = True
            skeleton[codes == 2] = 0
        if np.any(codes == 3): 
            pixelRemoved = True
            skeleton[codes == 3] = 0

    return skeleton
