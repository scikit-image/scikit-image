from __future__ import division, print_function, absolute_import

import numpy as np
from ..util import img_as_ubyte
from ._skeletonize_3d_cy import _compute_thin_image


def skeletonize_3d(img):
    """Compute the skeleton of a binary image.

    Thinning is used to reduce each connected component in a binary image
    to a single-pixel wide skeleton.

    Parameters
    ----------
    img : ndarray, 2D or 3D
        A binary image containing the objects to be skeletonized. Zeros
        represent background, nonzero values are foreground.

    Returns
    -------
    skeleton : ndarray
        The thinned image.

    See also
    --------
    skeletonize, medial_axis

    Notes
    -----
    The method of [Lee94]_ uses an octree data structure to examine a 3x3x3
    neighborhood of a pixel. The algorithm proceeds by iteratively sweeping
    over the image, and removing pixels at each iteration until the image
    stops changing. Each iteration consists of two steps: first, a list of
    candidates for removal is assembled; then pixels from this list are
    rechecked sequentially, to better preserve connectivity of the image.

    The algorithm this function implements is different from the algorithms
    used by either `skeletonize` or `medial_axis`, thus for 2D images the
    results produced by this function are generally different.

    References
    ----------
    .. [Lee94] T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models
           via 3-D medial surface/axis thinning algorithms.
           Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.

    """
    # make sure the image is 3D or 2D (if it is, temporarily upcast to 3D)
    if img.ndim < 2 or img.ndim > 3:
        raise ValueError('expect 2D, got ndim = %s' % img.ndim)

    img = img_as_ubyte(img)
    img = np.ascontiguousarray(img)

    img = img.copy()
    if img.ndim == 2:
        img = img[None, ...]

    # normalize to binary
    maxval = img.max()
    img[img != 0] = 1

    # pad w/ zeros to simplify dealing w/ boundaries
    img_o = np.pad(img, pad_width=1, mode='constant')

    # do the computation
    img_o = np.asarray(_compute_thin_image(img_o))

    # clip it back and restore the original intensity range
    img_o = img_o[1:-1, 1:-1, 1:-1]
    img_o = img_o.squeeze()
    img_o *= maxval

    return img_o
