from __future__ import division, print_function, absolute_import

import numpy as np
from ..util import image_as_ubyte, crop
from ._skeletonize_3d_cy import _compute_thin_image


def skeletonize_3d(image):
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
    # make sure the image is 3D or 2D
    if image.ndim < 2 or image.ndim > 3:
        raise ValueError("skeletonize_3d can only handle 2D or 3D images; "
                         "got image.ndim = %s instead." % image.ndim)

    image = np.ascontiguousarray(image)
    image = image_as_ubyte(image, force_copy=False)

    # make an in image 3D and pad it w/ zeros to simplify dealing w/ boundaries
    # NB: careful here to not clobber the original *and* minimize copying
    image_o = image
    if image.ndim == 2:
        image_o = image[np.newaxis, ...]
    image_o = np.pad(image_o, pad_width=1, mode='constant')

    # normalize to binary
    maxval = image_o.max()
    image_o[image_o != 0] = 1

    # do the computation
    image_o = np.asarray(_compute_thin_image(image_o))

    # crop it back and restore the original intensity range
    image_o = crop(image_o, crop_width=1)
    if image.ndim == 2:
        image_o = image_o[0]
    image_o *= maxval

    return image_o