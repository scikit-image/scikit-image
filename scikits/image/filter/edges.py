"""edges.py - Sobel edge filter

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky

"""
import numpy as np
from scipy.ndimage import convolve
from scikits.image.backend import add_backends
import sys


@add_backends
def sobel(image, axis=None, output=None):
    """Calculate the absolute magnitude Sobel to find the edges.

    Parameters
    ----------
    image : array_like, dtype=float
        Image to process
    mask : array_like, dtype=bool, optional
        An optional mask to limit the application to a certain area
    
    Returns
    -------
    output : ndarray
      The Sobel edge map.

    Notes
    -----
    Take the square root of the sum of the squares of the horizontal and
    vertical Sobels to get a magnitude that's somewhat insensitive to
    direction.
    
    Note that scipy's Sobel returns a directional Sobel which isn't
    useful for edge detection in its raw form.
    """
    print "running numpy sobel"
    if image.dtype == np.uint8:
        output_type = np.int16
    elif image.dtype == np.float32:
        output_type = np.float32
    if axis is None:
        dx = np.empty(image.shape, dtype=np.float32)
        dy = np.empty(image.shape, dtype=np.float32)
        convolve(image, np.array([[ 1, 2, 1],
                                  [ 0, 0, 0],
                                  [-1,-2,-1]]), output=dx)
        convolve(image, np.array([[ 1, 0,-1],
                                  [ 2, 0,-2],
                                  [ 1, 0,-1]]), output=dy)
        if output:
            output[:] = np.sqrt(dx ** 2 + dy ** 2)
            return output
        else:
            return np.sqrt(dx ** 2 + dy ** 2)
    elif axis == 0:
        dx = np.empty(image.shape, dtype=output_type)
        convolve(image, np.array([[ 1, 2, 1],
                                  [ 0, 0, 0],
                                  [-1,-2,-1]]), output=dx)
        return dx
    elif axis == 1:
        dy = np.empty(image.shape, dtype=output_type)
        convolve(image, np.array([[ 1, 0,-1],
                                  [ 2, 0,-2],
                                  [ 1, 0,-1]]), output=dy)
        return dy


    
#    hprewitt = np.abs(convolve(image, np.array([[ 1, 1, 1],
#                                              [ 0, 0, 0],
#                                              [-1,-1,-1]]).astype(float) / 3.0))                              
#    vprewitt = np.abs(convolve(image, np.array([[ 1, 0,-1],
#                                              [ 1, 0,-1],
#                                              [ 1, 0,-1]]).astype(float) / 3.0))

