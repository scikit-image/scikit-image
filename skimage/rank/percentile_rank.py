"""percentile_rank.py - inferior and superior ranks, provided by the user, are passed to the kernel function
to provide a softer version of the rank filters. E.g. percentile_autolevel will stretch image levels between
percentile [p0,p1] instead of using [min,max]. It means that isolate bright or dark pixels will not produce halos.

The local histogram is computed using a sliding window similar to the method described in

Reference: Huang, T. ,Yang, G. ;  Tang, G.. "A fast two-dimensional median filtering algorithm",
IEEE Transactions on Acoustics, Speech and Signal Processing, Feb 1979. Volume: 27 , Issue: 1, Page(s): 13 - 18.

input image can be 8 bit or 16 bit with a value < 4096 (i.e. 12 bit),
for 16 bit input images, the number of histogram bins is determined from the maximum value present in the image

result image is 8 or 16 bit with respect to the input image

:author: Olivier Debeir, 2012
:license: modified BSD
"""

__docformat__ = 'restructuredtext en'

import warnings
from skimage import img_as_ubyte
import numpy as np

from generic import find_bitdepth
import _crank16_percentiles,_crank8_percentiles

__all__ = ['percentile_autolevel','percentile_gradient',
           'percentile_mean','percentile_mean_substraction',
           'percentile_morph_contr_enh','percentile_pop']

def percentile_autolevel(image, selem, out=None, mask=None, shift_x=False, shift_y=False, p0=.0, p1=1.):
    """Return greyscale local autolevel of an image.

    Autolevel is computed on the given structuring element. Only levels between percentiles [p0,p1] ,are used.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, as the algorithm uses max. 12bit histogram,
        an exception will be raised if image has a value > 4095
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local neighborhood.
        If None, the complete image is used (default).
    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).
        Shift is bounded to the structuring element sizes.
    p0, p1 : float in [0.,...,1.]
        define the [p0,p1] percentile interval to be considered for computing the value.

    Returns
    -------
    local autolevel : uint8 array or uint16 array depending on input image
        The result of the local autolevel.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> percentile_autolevel(ima8, square(3), p0=0.,p1=1.)
    array([[  0,   0,   0,   0,   0],
           [  0, 255, 255, 255,   0],
           [  0, 255, 255, 255,   0],
           [  0, 255, 255, 255,   0],
           [  0,   0,   0,   0,   0]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> percentile_autolevel(ima16, square(3), p0=0.,p1=1.)
    array([[   0,    0,    0,    0,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0,    0,    0,    0,    0]], dtype=uint16)

    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8_percentiles.autolevel(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out,p0=p0,p1=p1)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16_percentiles.autolevel(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out,p0=p0,p1=p1)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def percentile_gradient(image, selem, out=None, mask=None, shift_x=False, shift_y=False, p0=.0, p1=1.):
    """Return greyscale local percentile_gradient of an image.

    percentile_gradient is computed on the given structuring element. Only levels between percentiles [p0,p1] ,are used.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, as the algorithm uses max. 12bit histogram,
        an exception will be raised if image has a value > 4095
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local neighborhood.
        If None, the complete image is used (default).
    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).
        Shift is bounded to the structuring element sizes.
    p0, p1 : float in [0.,...,1.]
        define the [p0,p1] percentile interval to be considered for computing the value.

    Returns
    -------
    local percentile_gradient : uint8 array or uint16 array depending on input image
        The result of the local percentile_gradient.

    Examples
    --------
    to be updated
    >>> # Local gradient
    >>> from skimage.morphology import square
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> percentile_gradient(ima8, square(3), p0=0.,p1=1.)
    array([[255, 255, 255, 255, 255],
           [255, 255, 255, 255, 255],
           [255, 255, 255, 255, 255],
           [255, 255, 255, 255, 255],
           [255, 255, 255, 255, 255]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> percentile_gradient(ima16, square(3), p0=0.,p1=1.)
    array([[4095, 4095, 4095, 4095, 4095],
           [4095, 4095, 4095, 4095, 4095],
           [4095, 4095, 4095, 4095, 4095],
           [4095, 4095, 4095, 4095, 4095],
           [4095, 4095, 4095, 4095, 4095]], dtype=uint16)

    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8_percentiles.gradient(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out,p0=p0,p1=p1)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16_percentiles.gradient(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out,p0=p0,p1=p1)
    else:
        raise TypeError("only uint8 and uint16 image supported!")


def percentile_mean(image, selem, out=None, mask=None, shift_x=False, shift_y=False, p0=.0, p1=1.):
    """Return greyscale local mean of an image.

    Mean is computed on the given structuring element. Only levels between percentiles [p0,p1] ,are used.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, as the algorithm uses max. 12bit histogram,
        an exception will be raised if image has a value > 4095
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local neighborhood.
        If None, the complete image is used (default).
    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).
        Shift is bounded to the structuring element sizes.
    p0, p1 : float in [0.,...,1.]
        define the [p0,p1] percentile interval to be considered for computing the value.

    Returns
    -------
    local mean : uint8 array or uint16 array depending on input image
        The result of the local mean.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> percentile_mean(ima8, square(3),p0=0.,p1=1.)
    array([[ 63,  85, 127,  85,  63],
           [ 85, 113, 170, 113,  85],
           [127, 170, 255, 170, 127],
           [ 85, 113, 170, 113,  85],
           [ 63,  85, 127,  85,  63]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> percentile_mean(ima16, square(3),p0=0.,p1=1.)
    array([[1023, 1365, 2047, 1365, 1023],
           [1365, 1820, 2730, 1820, 1365],
           [2047, 2730, 4095, 2730, 2047],
           [1365, 1820, 2730, 1820, 1365],
           [1023, 1365, 2047, 1365, 1023]], dtype=uint16)

    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8_percentiles.mean(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out,p0=p0,p1=p1)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16_percentiles.mean(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out,p0=p0,p1=p1)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def percentile_mean_substraction(image, selem, out=None, mask=None, shift_x=False, shift_y=False, p0=.0, p1=1.):
    """Return greyscale local mean_substraction of an image.

    mean_substraction is computed on the given structuring element. Only levels between percentiles [p0,p1] ,are used.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, as the algorithm uses max. 12bit histogram,
        an exception will be raised if image has a value > 4095
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local neighborhood.
        If None, the complete image is used (default).
    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).
        Shift is bounded to the structuring element sizes.
    p0, p1 : float in [0.,...,1.]
        define the [p0,p1] percentile interval to be considered for computing the value.

    Returns
    -------
    local mean_substraction : uint8 array or uint16 array depending on input image
        The result of the local mean_substraction.

    Examples
    --------
    to be updated
    >>> # Local mean_substraction
    >>> from skimage.morphology import square
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> percentile_mean_substraction(ima8, square(3), p0=0.,p1=1.)
    array([[ 95,  84,  63,  84,  95],
           [ 84, 198, 169, 198,  84],
           [ 63, 169, 127, 169,  63],
           [ 84, 198, 169, 198,  84],
           [ 95,  84,  63,  84,  95]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> percentile_mean_substraction(ima16, square(3), p0=0.,p1=1.)
    array([[1536, 1365, 1024, 1365, 1536],
           [1365, 3185, 2730, 3185, 1365],
           [1024, 2730, 2048, 2730, 1024],
           [1365, 3185, 2730, 3185, 1365],
           [1536, 1365, 1024, 1365, 1536]], dtype=uint16)

    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8_percentiles.mean_substraction(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out,p0=p0,p1=p1)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16_percentiles.mean_substraction(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out,p0=p0,p1=p1)
    else:
        raise TypeError("only uint8 and uint16 image supported!")


def percentile_morph_contr_enh(image, selem, out=None, mask=None, shift_x=False, shift_y=False, p0=.0, p1=1.):
    """Return greyscale local morph_contr_enh of an image.

    morph_contr_enh is computed on the given structuring element. Only levels between percentiles [p0,p1] ,are used.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, as the algorithm uses max. 12bit histogram,
        an exception will be raised if image has a value > 4095
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local neighborhood.
        If None, the complete image is used (default).
    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).
        Shift is bounded to the structuring element sizes.
    p0, p1 : float in [0.,...,1.]
        define the [p0,p1] percentile interval to be considered for computing the value.

    Returns
    -------
    local morph_contr_enh : uint8 array or uint16 array depending on input image
        The result of the local morph_contr_enh.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> percentile_morph_contr_enh(ima8, square(3), p0=0.,p1=1.)
    array([[  0,   0,   0,   0,   0],
           [  0, 255, 255, 255,   0],
           [  0, 255, 255, 255,   0],
           [  0, 255, 255, 255,   0],
           [  0,   0,   0,   0,   0]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> percentile_morph_contr_enh(ima16, square(3), p0=0.,p1=1.)
    array([[   0,    0,    0,    0,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0, 4095, 4095, 4095,    0],
           [   0,    0,    0,    0,    0]], dtype=uint16)

    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8_percentiles.morph_contr_enh(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out,p0=p0,p1=p1)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16_percentiles.morph_contr_enh(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out,p0=p0,p1=p1)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def percentile(image, selem, out=None, mask=None, shift_x=False, shift_y=False, p0=.0, p1=1.):
    """Return greyscale local percentile of an image.

    percentile is computed on the given structuring element. Only levels between percentiles [p0,p1] ,are used.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, as the algorithm uses max. 12bit histogram,
        an exception will be raised if image has a value > 4095
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local neighborhood.
        If None, the complete image is used (default).
    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).
        Shift is bounded to the structuring element sizes.
    p0, p1 : float in [0.,...,1.]
        define the [p0,p1] percentile interval to be considered for computing the value.

    Returns
    -------
    local percentile : uint8 array or uint16 array depending on input image
        The result of the local percentile.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> ima8 = 128*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> percentile(ima8, square(3), p0=0.,p1=1.)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> percentile(ima16, square(3), p0=0.,p1=1.)
    array([[0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0]], dtype=uint16)


    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8_percentiles.percentile(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out,p0=p0,p1=p1)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16_percentiles.percentile(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out,p0=p0,p1=p1)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def percentile_pop(image, selem, out=None, mask=None, shift_x=False, shift_y=False, p0=.0, p1=1.):
    """Return greyscale local pop of an image.

    pop is computed on the given structuring element. Only levels between percentiles [p0,p1] ,are used.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, as the algorithm uses max. 12bit histogram,
        an exception will be raised if image has a value > 4095
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local neighborhood.
        If None, the complete image is used (default).
    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).
        Shift is bounded to the structuring element sizes.
    p0, p1 : float in [0.,...,1.]
        define the [p0,p1] percentile interval to be considered for computing the value.

    Returns
    -------
    local pop : uint8 array or uint16 array depending on input image
        The result of the local pop.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> percentile_pop(ima8, square(3), p0=0.,p1=1.)
    array([[4, 6, 6, 6, 4],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [4, 6, 6, 6, 4]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> percentile_pop(ima16, square(3), p0=0.,p1=1.)
    array([[4, 6, 6, 6, 4],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [6, 9, 9, 9, 6],
           [4, 6, 6, 6, 4]], dtype=uint16)

    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8_percentiles.pop(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out,p0=p0,p1=p1)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16_percentiles.pop(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out,p0=p0,p1=p1)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

def percentile_threshold(image, selem, out=None, mask=None, shift_x=False, shift_y=False, p0=.0, p1=1.):
    """Return greyscale local threshold of an image.

    threshold is computed on the given structuring element. Only levels between percentiles [p0,p1] ,are used.

    Parameters
    ----------
    image : ndarray
        Image array (uint8 array or uint16). If image is uint16, as the algorithm uses max. 12bit histogram,
        an exception will be raised if image has a value > 4095
    selem : ndarray
        The neighborhood expressed as a 2-D array of 1's and 0's.
    out : ndarray
        The array to store the result of the morphology. If None is
        passed, a new array will be allocated.
    mask : ndarray (uint8)
        Mask array that defines (>0) area of the image included in the local neighborhood.
        If None, the complete image is used (default).
    shift_x, shift_y : bool
        shift structuring element about center point. This only affects
        eccentric structuring elements (i.e. selem with even numbered sides).
        Shift is bounded to the structuring element sizes.
    p0, p1 : float in [0.,...,1.]
        define the [p0,p1] percentile interval to be considered for computing the value.

    Returns
    -------
    local threshold : uint8 array or uint16 array depending on input image
        The result of the local threshold.

    Examples
    --------
    to be updated
    >>> # Local mean
    >>> from skimage.morphology import square
    >>> ima8 = 255*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> percentile_threshold(ima8, square(3), p0=0.,p1=1.)
    array([[255, 255, 255, 255, 255],
           [255, 255, 255, 255, 255],
           [255, 255, 255, 255, 255],
           [255, 255, 255, 255, 255],
           [255, 255, 255, 255, 255]], dtype=uint8)

    >>> ima16 = 4095*np.array([[0, 0, 0, 0, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 1, 1, 1, 0],
    ...                           [0, 0, 0, 0, 0]], dtype=np.uint16)
    >>> percentile_threshold(ima16, square(3), p0=0.,p1=1.)
    array([[4095, 4095, 4095, 4095, 4095],
           [4095, 4095, 4095, 4095, 4095],
           [4095, 4095, 4095, 4095, 4095],
           [4095, 4095, 4095, 4095, 4095],
           [4095, 4095, 4095, 4095, 4095]], dtype=uint16)


    """
    selem = img_as_ubyte(selem)
    if mask is not None:
        mask = img_as_ubyte(mask)
    if image.dtype == np.uint8:
        return _crank8_percentiles.threshold(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,out=out,p0=p0,p1=p1)
    elif image.dtype == np.uint16:
        bitdepth = find_bitdepth(image)
        if bitdepth>11:
            raise ValueError("only uint16 <4096 image (12bit) supported!")
        return _crank16_percentiles.threshold(image,selem,shift_x=shift_x,shift_y=shift_y,mask=mask,bitdepth=bitdepth+1,out=out,p0=p0,p1=p1)
    else:
        raise TypeError("only uint8 and uint16 image supported!")

