""" A 2D haar feature detector"""

import numpy as np
import scipy.ndimage as nd
import math

from skimage.transform.haar2d import haar2d

def haar(image, levels=2, threshold=.2, locality=5):
    """
    2D Haar wavelet feature detector. The levels of the Haar decomposition
    are equally weigted, so features are salient either if there is change on
    many levels, or very large change on some level. The maximum level that
    can be specified for an image depends on the dimensions of the image: 
    Each image dimension divided by 2 ** N must be >= 1 where N is the number 
    of levels chosen.
    
    Parameters
    ----------
    image: nd-array
        Input image 
    levels: int
        Number of wavelet levels to compute
    threshold: float
        Threshold level relative to maximum feature saliency
    locality: int
        Minimum distance between features

    Returns
    -------
    features: ndarray
       An array containing the coordinates [[row,col], ..] of the detected features.

    See also
    --------
    skimage.transform.haar2d
    """
    
    haarData = haar2d(image, levels)

    avgRows = haarData.shape[0] / 2 ** levels
    avgCols = haarData.shape[1] / 2 ** levels
    
    SalientPoints = {}    
    siloH = np.zeros([haarData.shape[0]/2, haarData.shape[1]/2, levels])
    siloD = np.zeros([haarData.shape[0]/2, haarData.shape[1]/2, levels])
    siloV = np.zeros([haarData.shape[0]/2, haarData.shape[1]/2, levels])
    
    # Build the saliency 'silos'
    for i in range(levels):
        level = i + 1
        halfRows = haarData.shape[0] / 2 ** level
        halfCols = haarData.shape[1] / 2 ** level
        siloH[:,:,i] = nd.zoom(haarData[:halfRows, halfCols:halfCols*2], 2**(level-1)) 
        siloD[:,:,i] = nd.zoom(haarData[halfRows:halfRows*2, halfCols:halfCols*2], 2**(level-1)) 
        siloV[:,:,i] = nd.zoom(haarData[halfRows:halfRows*2, :halfCols], 2**(level-1)) 
    
    # Calculate saliency 'heat-map'
    saliencyMap = np.max(np.array([
                                np.sum(np.abs(siloH), axis=2), 
                                np.sum(np.abs(siloD), axis=2),
                                np.sum(np.abs(siloV), axis=2)
                                ]), axis=0)
                               
    # Determine global maximum and saliency threshold
    maximum = np.max(saliencyMap)
    sthreshold = threshold * maximum
    
    # Extract features by finding local maxima
    rows = haarData.shape[0] / 2
    cols = haarData.shape[1] / 2
    features = []
    for row in range(locality,rows-locality):
        for col in range(locality,cols-locality):
            saliency = saliencyMap[row,col]
            if saliency > sthreshold:
                if  saliency >= np.max(saliencyMap[row-locality:row+locality, col-locality:col+locality]):
                    features.append([row*2,col*2])

    return np.array(features)
    



