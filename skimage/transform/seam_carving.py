from _seam_carving import _seam_carve_v
from ..import filters
from .. import util
from .._shared import utils
import numpy as np


def seam_carve(img, mode, num, energy_func, extra_args = [],
               extra_kwargs = {}, border=1, force_copy = True):
    """ Carve vertical or horizontal seams off an image.

    Carves out vertical/horizontal seams off an image while using the given
    energy function to decide the importance of each pixel.

    Parameters
    ----------
    image : (M, N) or (M, N, 3) ndarray
        Input image whose vertical seams are to be removed.
    mode : str {'horizontal', 'vertical'}
        Indicates whether seams are to be removed vertically or horizontally.
        Removing seams horizontally will decrease the height whereas removing
        vertically will decrease the width.
    num : int
        Number of seams are to be removed.
    energy_func : callable
        The function used to decide the importance of each pixel. The higher
        the value corresponding to a pixel, the more the algorithm will try
        to keep it in the image. For every iteration `energy_func` is called
        as `energy_func(image, *extra_args, **extra_kwargs)`, where `image`
        is the cropped image during each iteration and is expected to return a
        (M, N) ndarray depicting each pixel's importance.
    extra_args : iterable, optional
        The extra arguments supplied to `energy_func`.
    extra_kwargs : dict, optional
        The extra keyword arguments supplied to `energy_func`.
    border : int, optional
        The number of pixels in the right and left end of the image to be
        excluded from being considered for a seam. This is important as certain
        filters just ignore image boundaries and set them to `0`. By default
        border is set to `1`.
    force_copy : bool, optional
        If set, the image is copied before being used by the method which
        modifies it in place. Set this to `False` if the original image is no
        loner needed after this opetration.

    Returns
    -------
    out : ndarray
        The cropped image with the seams removed.

    References
    ----------
    .. [1] Shai Avidan and Ariel Shamir
           "Seam Carving for Content-Aware Image Resizing"
           http://www.cs.jhu.edu/~misha/ReadingSeminar/Papers/Avidan07.pdf
    """

    utils.assert_nD(img, (2,3))
    img = util.img_as_float(img)
    

    if mode == 'horizontal':
        img = np.ascontiguousarray(img)
        return _seam_carve_v(img, num, energy_func, extra_args ,extra_kwargs,
                             border)
    elif mode == 'vertical' :
        if img.ndim == 3:
            img = np.transpose(img, (1, 0, 2))
        else:
            img = img.T

        img = np.ascontiguousarray(img)
        out = _seam_carve_v(img, num, energy_func, extra_args , extra_kwargs,
                            border)

        if img.ndim == 3:
            return  np.transpose(out, (1, 0, 2))
        else:
            return out.T
