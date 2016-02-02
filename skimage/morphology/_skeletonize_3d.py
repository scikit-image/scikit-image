from __future__ import division, print_function, absolute_import

import numpy as np
from ._skeletonize_3d_cy import _compute_thin_image


def skeletonize_3d(img_in):
    """Compute the skeleton of a binary image.

    Thinning is used to reduce each connected component in a binary image
    to a single-pixel wide skeleton.

    Parameters
    ----------
    image : ndarray, 2D or 3D
        A binary image containing the objects to be skeletonized. Zeros
        represent background, nonzero values are foreground.

    Returns
    -------
    skeleton : ndarray
        The thinned image.

    See also
    --------
    skeletonize, medial_axis

    References
    ----------
    .. [Lee94] Lee et al, Building skeleton models via 3-D medial surface/axis
           thinning algorithms. Computer Vision, Graphics, and Image Processing,
           56(6):462–478, 1994.

    """
    # make sure the image is 3D or 2D (if it is, temporarily upcast to 3D)
    if img_in.ndim < 2 or img_in.ndim > 3:
        raise ValueError('expect 2D, got ndim = %s' % img_in.ndim)

    img = img_in.copy()
    if img.ndim == 2:
        img = img[None, ...]

    # normalize to binary
    maxval = img.max()
    img[img != 0] = 1
    img = img.astype(np.uint8)

    # pad w/ zeros to simplify dealing w/ boundaries
    img_o = np.pad(img, pad_width=1, mode='constant')

    # do the computation
    img_o = np.asarray(_compute_thin_image(img_o))

    # clip it back and restore the original intensity range
    img_o = img_o[1:-1, 1:-1, 1:-1]
    img_o = img_o.squeeze()
    img_o *= maxval

    return img_o
