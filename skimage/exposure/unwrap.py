import numpy as np

from ._unwrap_2d import unwrap_2d
from ._unwrap_3d import unwrap_3d
from .._shared.six import string_types


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
           path", Journal Applied Optics, Vol. 41, No. 35 (2002) 7437,
    .. [2] Abdul-Rahman, H., Gdeisat, M., Burton, D., & Lalor, M., "Fast
           three-dimensional phase-unwrapping algorithm based on sorting by
           reliability following a non-continuous path. In W. Osten,
           C. Gorecki, & E. L. Novak (Eds.), Optical Metrology (2005) 32--40,
           International Society for Optics and Photonics.
    '''
    if image.ndim not in (2, 3):
        raise ValueError('image must be 2 or 3 dimensional')
    if isinstance(wrap_around, bool):
        wrap_around = [wrap_around] * image.ndim
    elif (hasattr(wrap_around, '__getitem__')
          and not isinstance(wrap_around, string_types)):
        if not len(wrap_around) == image.ndim:
            raise ValueError('Length of wrap_around must equal the '
                             'dimensionality of image')
        wrap_around = [bool(wa) for wa in wrap_around]
    else:
        raise ValueError('wrap_around must be a bool or a sequence with '
                         'length equal to the dimensionality of image')

    if np.ma.isMaskedArray(image):
        mask = np.require(image.mask, np.uint8, ['C'])
    else:
        mask = np.zeros(image.shape, dtype=np.uint8, order='C')
    image_not_masked = np.asarray(image, dtype=np.float32, order='C')
    image_unwrapped = np.empty(image.shape, dtype=np.float32)

    if image.ndim == 2:
        unwrap_2d(image_not_masked, mask, image_unwrapped,
                  wrap_around)
    elif image.ndim == 3:
        unwrap_3d(image_not_masked, mask, image_unwrapped,
                  wrap_around)

    if np.ma.isMaskedArray(image):
        return np.ma.array(image_unwrapped, mask=mask)
    else:
        return image_unwrapped
