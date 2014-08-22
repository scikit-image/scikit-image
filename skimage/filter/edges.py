"""

Sobel and Prewitt filters originally part of CellProfiler, code licensed under
both GPL and BSD licenses.
Website: http://www.cellprofiler.org
Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.
Original author: Lee Kamentsky

"""
import numpy as np
from skimage import img_as_float
from scipy.ndimage import convolve, binary_erosion, generate_binary_structure


EROSION_SELEM = generate_binary_structure(2, 2)

HSOBEL_WEIGHTS = np.array([[ 1, 2, 1],
                           [ 0, 0, 0],
                           [-1,-2,-1]]) / 4.0
VSOBEL_WEIGHTS = HSOBEL_WEIGHTS.T

HSCHARR_WEIGHTS = np.array([[ 3,  10,  3],
                            [ 0,   0,  0],
                            [-3, -10, -3]]) / 16.0
VSCHARR_WEIGHTS = HSCHARR_WEIGHTS.T

HPREWITT_WEIGHTS = np.array([[ 1, 1, 1],
                             [ 0, 0, 0],
                             [-1,-1,-1]]) / 3.0
VPREWITT_WEIGHTS = HPREWITT_WEIGHTS.T

ROBERTS_PD_WEIGHTS = np.array([[1, 0],
                               [0, -1]], dtype=np.double)
ROBERTS_ND_WEIGHTS = np.array([[0, 1],
                               [-1, 0]], dtype=np.double)


def _mask_filter_result(result, mask):
    """Return result after masking.

    Input masks are eroded so that mask areas in the original image don't
    affect values in the result.
    """
    if mask is None:
        result[0, :] = 0
        result[-1, :] = 0
        result[:, 0] = 0
        result[:, -1] = 0
        return result
    else:
        mask = binary_erosion(mask, EROSION_SELEM, border_value=0)
        return result * mask


def sobel(image, mask=None):
    """Find the edge magnitude using the Sobel transform.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Sobel edge map.

    Notes
    -----
    Take the square root of the sum of the squares of the horizontal and
    vertical Sobels to get a magnitude that's somewhat insensitive to
    direction.

    Note that ``scipy.ndimage.sobel`` returns a directional Sobel which
    has to be further processed to perform edge detection.
    """
    return np.sqrt(hsobel(image, mask)**2 + vsobel(image, mask)**2)


def hsobel(image, mask=None):
    """Find the horizontal edges of an image using the Sobel transform.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
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
    result = np.abs(convolve(image, HSOBEL_WEIGHTS))
    return _mask_filter_result(result, mask)


def vsobel(image, mask=None):
    """Find the vertical edges of an image using the Sobel transform.

    Parameters
    ----------
    image : 2-D array
        Image to process
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
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
    result = np.abs(convolve(image, VSOBEL_WEIGHTS))
    return _mask_filter_result(result, mask)


def scharr(image, mask=None):
    """Find the edge magnitude using the Scharr transform.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Scharr edge map.

    Notes
    -----
    Take the square root of the sum of the squares of the horizontal and
    vertical Scharrs to get a magnitude that's somewhat insensitive to
    direction.

    References
    ----------
    .. [1] D. Kroon, 2009, Short Paper University Twente, Numerical Optimization
           of Kernel Based Image Derivatives.

    """
    return np.sqrt(hscharr(image, mask)**2 + vscharr(image, mask)**2)


def hscharr(image, mask=None):
    """Find the horizontal edges of an image using the Scharr transform.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Scharr edge map.

    Notes
    -----
    We use the following kernel and return the absolute value of the
    result at each point::

      3   10   3
      0    0   0
     -3  -10  -3

    References
    ----------
    .. [1] D. Kroon, 2009, Short Paper University Twente, Numerical Optimization
           of Kernel Based Image Derivatives.

    """
    image = img_as_float(image)
    result = np.abs(convolve(image, HSCHARR_WEIGHTS))
    return _mask_filter_result(result, mask)


def vscharr(image, mask=None):
    """Find the vertical edges of an image using the Scharr transform.

    Parameters
    ----------
    image : 2-D array
        Image to process
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Scharr edge map.

    Notes
    -----
    We use the following kernel and return the absolute value of the
    result at each point::

       3   0   -3
      10   0  -10
       3   0   -3

    References
    ----------
    .. [1] D. Kroon, 2009, Short Paper University Twente, Numerical Optimization
           of Kernel Based Image Derivatives.

    """
    image = img_as_float(image)
    result = np.abs(convolve(image, VSCHARR_WEIGHTS))
    return _mask_filter_result(result, mask)


def prewitt(image, mask=None):
    """Find the edge magnitude using the Prewitt transform.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Prewitt edge map.

    Notes
    -----
    Return the square root of the sum of squares of the horizontal
    and vertical Prewitt transforms.
    """
    return np.sqrt(hprewitt(image, mask)**2 + vprewitt(image, mask)**2)


def hprewitt(image, mask=None):
    """Find the horizontal edges of an image using the Prewitt transform.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
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
    result = np.abs(convolve(image, HPREWITT_WEIGHTS))
    return _mask_filter_result(result, mask)


def vprewitt(image, mask=None):
    """Find the vertical edges of an image using the Prewitt transform.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
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
    result = np.abs(convolve(image, VPREWITT_WEIGHTS))
    return _mask_filter_result(result, mask)


def roberts(image, mask=None):
    """Find the edge magnitude using Roberts' cross operator.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Roberts' Cross edge map.
    """
    return np.sqrt(roberts_positive_diagonal(image, mask)**2 +
                   roberts_negative_diagonal(image, mask)**2)


def roberts_positive_diagonal(image, mask=None):
    """Find the cross edges of an image using Roberts' cross operator.

    The kernel is applied to the input image to produce separate measurements
    of the gradient component one orientation.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Robert's edge map.

    Notes
    -----
    We use the following kernel and return the absolute value of the
    result at each point::

      1   0
      0  -1

    """
    image = img_as_float(image)
    result = np.abs(convolve(image, ROBERTS_PD_WEIGHTS))
    return _mask_filter_result(result, mask)


def roberts_negative_diagonal(image, mask=None):
    """Find the cross edges of an image using the Roberts' Cross operator.

    The kernel is applied to the input image to produce separate measurements
    of the gradient component one orientation.

    Parameters
    ----------
    image : 2-D array
        Image to process.
    mask : 2-D array, optional
        An optional mask to limit the application to a certain area.
        Note that pixels surrounding masked regions are also masked to
        prevent masked regions from affecting the result.

    Returns
    -------
    output : 2-D array
        The Robert's edge map.

    Notes
    -----
    We use the following kernel and return the absolute value of the
    result at each point::

      0   1
     -1   0

    """
    image = img_as_float(image)
    result = np.abs(convolve(image, ROBERTS_ND_WEIGHTS))
    return _mask_filter_result(result, mask)
