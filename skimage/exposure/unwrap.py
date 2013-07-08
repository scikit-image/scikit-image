import numpy as np

import _unwrap_2d
import _unwrap_3d


def unwrap(image, wrap_around=False):
    '''From ``image``, wrapped to lie in the interval [-pi, pi), recover the
    original, unwrapped image.

    Parameters
    ----------
    image : 2D or 3D ndarray, optionally a masked array
    wrap_around : bool or sequence of bool

    Returns
    -------
    image_unwrapped : array_like
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

