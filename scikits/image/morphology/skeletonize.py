"""skeletonize.py - ???

Original author: Neil Yager
"""

import numpy as np
from scipy.ndimage import correlate

def skeletonize(image):
    """
    Return a single pixel wide skeleton of all connected
     components in a binary image

    Parameters
    ----------
    
    image:
    
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
    
    skeleton = image.copy().astype(np.int8)
    mask = np.array([[  1,  2,  4],
                     [128,  0,  8],
                     [ 64, 32, 16]], np.int8)

    pixelRemoved = True
    while pixelRemoved:
        pixelRemoved = False;

        # pass 1
        neighbours = correlate(skeleton, mask, mode='constant')
        neighbours[skeleton == 0] = 0
        codes = np.take(lut, neighbours)
        if np.any(codes == 1): 
            pixelRemoved = True
            skeleton[codes == 1] = 0
        if np.any(codes == 3): 
            pixelRemoved = True
            skeleton[codes == 3] = 0

        # pass 2
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

    
             