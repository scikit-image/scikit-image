"""edges.py - Sobel edge filter

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky

"""
import numpy as np
from skimage import img_as_float
from scipy.ndimage import convolve, binary_erosion, generate_binary_structure


def sobel(image, mask=None):
    """Calculate the absolute magnitude Sobel to find edges.

    Parameters
    ----------
    image : array_like, dtype=float
        Image to process.
    mask : array_like, dtype=bool, optional
        An optional mask to limit the application to a certain area.

    Returns
    -------
    output : ndarray
      The Sobel edge map.

    Notes
    -----
    Take the square root of the sum of the squares of the horizontal and
    vertical Sobels to get a magnitude that's somewhat insensitive to
    direction.

    Note that ``scipy.ndimage.sobel`` returns a directional Sobel which
    has to be further processed to perform edge detection.
    """
    return np.sqrt(hsobel(image, mask) ** 2 + vsobel(image, mask) ** 2)


def hsobel(image, mask=None):
    """Find the horizontal edges of an image using the Sobel transform.

    Parameters
    ----------
    image : array_like, dtype=float
        Image to process.
    mask : array_like, dtype=bool, optional
        An optional mask to limit the application to a certain area.

    Returns
    -------
    output : ndarray
      The Sobel edge map.

    Notes
    -----
    We use the following kernel and return the absolute value of the
    result at each point::

      1   2   1
      0   0   0
     -1  -2  -1

    """
    image = img_as_float(image)
    if mask is None:
        mask = np.ones(image.shape, bool)
    big_mask = binary_erosion(mask,
                              generate_binary_structure(2, 2),
                              border_value = 0)
    result = np.abs(convolve(image,
                             np.array([[ 1, 2, 1],
                                       [ 0, 0, 0],
                                       [-1,-2,-1]]).astype(float) / 4.0))
    result[big_mask == False] = 0
    return result

def vsobel(image, mask=None):
    """Find the vertical edges of an image using the Sobel transform.

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
    We use the following kernel and return the absolute value of the
    result at each point::

      1   0  -1
      2   0  -2
      1   0  -1

    """
    image = img_as_float(image)
    if mask is None:
        mask = np.ones(image.shape, bool)
    big_mask = binary_erosion(mask,
                              generate_binary_structure(2, 2),
                              border_value=0)
    result = np.abs(convolve(image,
                             np.array([[1, 0, -1],
                                       [2, 0, -2],
                                       [1, 0, -1]]).astype(float) / 4.0))
    result[big_mask == False] = 0
    return result


def prewitt(image, mask=None):
    """Find the edge magnitude using the Prewitt transform.

    Parameters
    ----------
    image : array_like, dtype=float
        Image to process.
    mask : array_like, dtype=bool, optional
        An optional mask to limit the application to a certain area.

    Returns
    -------
    output : ndarray
      The Prewitt edge map.

    Notes
    -----
    Return the square root of the sum of squares of the horizontal
    and vertical Prewitt transforms.
    """
    return np.sqrt(hprewitt(image, mask) ** 2 + vprewitt(image, mask) ** 2)


def hprewitt(image, mask=None):
    """Find the horizontal edges of an image using the Prewitt transform.

    Parameters
    ----------
    image : array_like, dtype=float
        Image to process.
    mask : array_like, dtype=bool, optional
        An optional mask to limit the application to a certain area.

    Returns
    -------
    output : ndarray
      The Prewitt edge map.

    Notes
    -----
    We use the following kernel and return the absolute value of the
    result at each point::

      1   1   1
      0   0   0
     -1  -1  -1

    """
    image = img_as_float(image)
    if mask is None:
        mask = np.ones(image.shape, bool)
    big_mask = binary_erosion(mask,
                              generate_binary_structure(2, 2),
                              border_value=0)
    result = np.abs(convolve(image,
                             np.array([[ 1, 1, 1],
                                       [ 0, 0, 0],
                                       [-1,-1,-1]]).astype(float) / 3.0))
    result[big_mask == False] = 0
    return result


def vprewitt(image, mask=None):
    """Find the vertical edges of an image using the Prewitt transform.

    Parameters
    ----------
    image : array_like, dtype=float
        Image to process.
    mask : array_like, dtype=bool, optional
        An optional mask to limit the application to a certain area.

    Returns
    -------
    output : ndarray
      The Prewitt edge map.

    Notes
    -----
    We use the following kernel and return the absolute value of the
    result at each point::

      1   0  -1
      1   0  -1
      1   0  -1

    """
    image = img_as_float(image)
    if mask is None:
        mask = np.ones(image.shape, bool)
    big_mask = binary_erosion(mask,
                              generate_binary_structure(2, 2),
                              border_value=0)
    result = np.abs(convolve(image,
                             np.array([[1, 0, -1],
                                       [1, 0, -1],
                                       [1, 0, -1]]).astype(float) / 3.0))
    result[big_mask == False] = 0
    return result
