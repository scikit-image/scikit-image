import numpy as np
import scipy.ndimage as nd
import math

def haar2d(image, levels=1):
    """
    2D Haar wavelet decomposition for levels=levels.

    Please refer to http://en.wikipedia.org/wiki/Wavelets for a description
    of what a 'level' is.
    
    Parameters
    ----------
    image: nd-array
        Input image 
    levels: int
        Number of wavelet levels to compute

    Returns
    -------
    haarImage: nd-array
       An image containing the Haar decomposition of the input image. 
       Might be larger than the input image.

    See also
    --------
    skimage.transfrom.ihaar2d
    """
    if image.ndim != 2:
        raise ValueError('The input image must be 2-D')
    origRows, origCols = image.shape
    extraRows = 0;
    extraCols = 0;
    while np.left_shift(np.right_shift((origRows + extraRows), levels), levels) != (origRows + extraRows):
        extraRows += 1
    while np.left_shift(np.right_shift((origCols + extraCols), levels), levels) != (origCols + extraCols):
        extraCols += 1

    # Pad image to compatible shape using repetition
    rightFill = np.repeat(image[:, -1:], extraCols, axis=1)
    _image = np.zeros([origRows, origCols + extraCols])
    _image[:, :origCols] = image
    _image[:, origCols:] = rightFill
    bottomFill = np.repeat(_image[-1:, :], extraRows, axis=0)
    image = np.zeros([origRows + extraRows, origCols + extraCols])
    image[:origRows, :] = _image
    image[origRows:, :] = bottomFill

    haarImage = image
    for level in range(1,levels+1):
        halfRows = image.shape[0] / 2 ** level
        halfCols = image.shape[1] / 2 ** level
        _image = image[:halfRows*2, :halfCols*2]
        # rows
        lowpass = (_image[:, :-1:2] + _image[:, 1::2]) / 2
        higpass = (_image[:, :-1:2] - _image[:, 1::2]) / 2
        _image[:, :_image.shape[1]/2] = lowpass
        _image[:, _image.shape[1]/2:] = higpass
        # cols
        lowpass = (_image[:-1:2, :] + _image[1::2, :]) / 2
        higpass = (_image[:-1:2, :] - _image[1::2, :]) / 2
        _image[:_image.shape[0]/2, :] = lowpass
        _image[_image.shape[0]/2:, :] = higpass
        haarImage[:halfRows*2, :halfCols*2] = _image    

    return haarImage

def ihaar2d(image, levels=1):
    """
    2D Haar wavelet decomposition inverse for levels=levels.
    
    Please refer to http://en.wikipedia.org/wiki/Wavelets for a description
    of what a 'level' is.

    Parameters
    ----------
    image: nd-array
        Input image 
    levels: int
        Number of wavelet levels to de-compute

    Returns
    -------
    image: nd-array
       An image containing the inverse Haar decomposition of the input image. 

    See also
    --------
    skimage.transform.haar2d
    """    
    if image.ndim != 2:
      raise ValueError('The input image must be 2-D')
    
    origRows, origCols = image.shape
    extraRows = 0;
    extraCols = 0;
    while np.left_shift(np.right_shift((origRows + extraRows), levels), levels) != (origRows + extraRows):
        extraRows += 1
    while np.left_shift(np.right_shift((origCols + extraCols), levels), levels) != (origCols + extraCols):
        extraCols += 1
    assert (extraRows, extraCols) == (0,0), 'Must be compatible shape!'

    for level in range(levels, 0, -1):
        halfRows = image.shape[0] / 2 ** level
        halfCols = image.shape[1] / 2 ** level
        # cols
        lowpass = image[:halfRows*2, :halfCols].copy()
        higpass = image[:halfRows*2, halfCols:halfCols*2].copy()
        image[:halfRows*2, :halfCols*2-1:2] = lowpass + higpass 
        image[:halfRows*2, 1:halfCols*2:2] = lowpass - higpass
        # rows
        lowpass = image[:halfRows, :halfCols*2].copy()
        higpass = image[halfRows:halfRows*2, :halfCols*2].copy()
        image[:halfRows*2-1:2, :halfCols*2] = lowpass + higpass
        image[1:halfRows*2:2, :halfCols*2] = lowpass - higpass

    return image
