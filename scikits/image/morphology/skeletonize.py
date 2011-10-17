"""skeletonize.py - Use an iterative thinning algorithm to find the
                    skeletons of binary objects in an image.
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
    axis transformation, which can be can be implemented using  
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
    
    # look up table - there is one entry for each of the 2^8=256 possible
    # combinations of 8 binary neighbours. 1's, 2's and 3's are candidates
    # for removal at each iteration of the algorithm.
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
    if not np.all(np.in1d(skeleton.flat, (0, 1))):
        raise ValueError('Image contains values other than 0 and 1')
    
    # create the mask that will assign a unique value based on the
    #  arrangement of neighbouring pixels
    mask = np.array([[  1,  2,  4],
                     [128,  0,  8],
                     [ 64, 32, 16]], np.uint8)

    pixelRemoved = True
    while pixelRemoved:
        pixelRemoved = False;

        # assign each pixel a unique value based on its foreground neighbours
        neighbours = correlate(skeleton, mask, mode='constant')
        
        # ignore background
        neighbours[skeleton == 0] = 0
        
        # use LUT to categorize each foreground pixel as a 0, 1, 2 or 3
        codes = np.take(lut, neighbours)
        
        # pass 1 - remove the 1's and 3's
        code_mask = (codes == 1)
        if np.any(code_mask): 
            pixelRemoved = True
            skeleton[code_mask] = 0
        code_mask = (codes == 3)
        if np.any(code_mask): 
            pixelRemoved = True
            skeleton[code_mask] = 0

        # pass 2 - remove the 2's and 3's
        neighbours = correlate(skeleton, mask, mode='constant')
        neighbours[skeleton == 0] = 0
        codes = np.take(lut, neighbours)
        code_mask = (codes == 2)
        if np.any(code_mask): 
            pixelRemoved = True
            skeleton[code_mask] = 0
        code_mask = (codes == 3)
        if np.any(code_mask): 
            pixelRemoved = True
            skeleton[code_mask] = 0

    return skeleton
