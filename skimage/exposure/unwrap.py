import numpy as np

from . import _unwrap_2d
from . import _unwrap_3d


def unwrap(image, wrap_around=False):
    '''From ``image``, wrapped to lie in the interval [-pi, pi), recover the
    original, unwrapped image.

    Parameters
    ----------
    image : 2D or 3D ndarray of floats, optionally a masked array
        The values should be in the range ``[-pi, pi)``. If a masked array is
        provided, the masked entries will not be changed, and their values
        will not be used to guide the unwrapping of neighboring, unmasked
        values.
    wrap_around : bool or sequence of bool
        When an element of the sequence is  ``True``, the unwrapping process
        will regard the edges along the corresponding axis of the image to be
        connected and use this connectivity to guide the phase unwrapping
        process. If only a single boolean is given, it will apply to all axes.

    Returns
    -------
    image_unwrapped : array_like, float32
        Unwrapped image of the same shape as the input. If the input ``image``
        was a masked array, the mask will be preserved.

    References
    ----------
    .. [1] Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor,
           and Munther A. Gdeisat, "Fast two-dimensional phase-unwrapping
           algorithm based on sorting by reliability following a noncontinuous
           path", Journal Applied Optics, Vol. 41, No. 35, pp. 7437, 2002
    '''
    image = np.require(image, np.float32, ['C'])
    if image.ndim not in (2, 3):
        raise ValueError('image must be 2 or 3 dimensional')
    if isinstance(wrap_around, bool):
        wrap_around = [wrap_around] * image.ndim
    elif (hasattr(wrap_around, '__getitem__')
          and not isinstance(wrap_around, basestring)):
        if not len(wrap_around) == image.ndim:
            raise ValueError('Length of wrap_around must equal the '
                             'dimensionality of image')
        wrap_around = [bool(wa) for wa in wrap_around]
    else:
        raise ValueError('wrap_around must be a bool or a sequence with '
                         'length equal to the dimensionality of image')

    image_masked = np.ma.asarray(image)
    image_unwrapped = np.empty_like(image_masked.data)
    if image.ndim == 2:
        _unwrap_2d._unwrap2D(image_masked.data,
                           np.ma.getmaskarray(image_masked).astype(np.uint8),
                           image_unwrapped,
                           wrap_around[0], wrap_around[1])
    elif image.ndim == 3:
        _unwrap_3d._unwrap3D(image_masked.data,
                           np.ma.getmaskarray(image_masked).astype(np.uint8),
                           image_unwrapped,
                           wrap_around[0], wrap_around[1], wrap_around[2])

    if np.ma.isMaskedArray(image):
        return np.ma.array(image_unwrapped, mask = image_masked.mask)
    else:
        return image_unwrapped

    #TODO: set_fill to minimum value
    #TODO: check for empty mask, not a single contiguous pixel

