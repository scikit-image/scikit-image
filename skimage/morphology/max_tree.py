"""max_tree.py - max_tree representation of images.

This module provides operators based on the max-tree representation of images.
A grey scale image can be seen as a pile of nested sets, each of which is the
result of a threshold operation. These sets can be efficiently represented by
max-trees, where the inclusion relation between connected components at
different levels are represented by parent-child relationships.

These representations allow efficient implementations of many algorithms, such
as attribute operators. Unlike morphological openings and closings, these
operators do not require a fixed structuring element, but rather act with a
flexible structuring element that meets a certain criterion.

This implementation provides functions for:
1. max-tree generation
2. area openings / closings
3. diameter openings / closings
4. local maxima / minima
5. extended local maxima / minima
6. h-maxima / h-minima (corresponding to dynamics)

References:
    .. [1] Salembier, P., Oliveras, A., & Garrido, L. (1998). Antiextensive
           Connected Operators for Image and Sequence Processing.
           IEEE Transactions on Image Processing, 7(4), 555-570.
    .. [2] Berger, C., Geraud, T., Levillain, R., Widynski, N., Baillard, A.,
           Bertin, E. (2007). Effective Component Tree Computation with
           Application to Pattern Recognition in Astronomical Imaging.
           In International Conference on Image Processing (ICIP) (pp. 41-44).
    .. [3] Najman, L., & Couprie, M. (2006). Building the component tree in
           quasi-linear time. IEEE Transactions on Image Processing, 15(11),
           3531-3539.
    .. [4] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
           Component Tree Computation Algorithms. IEEE Transactions on Image
           Processing, 23(9), 3885-3895.
"""

import numpy as np

from .watershed import _validate_connectivity
from .watershed import _compute_neighbors

from . import _max_tree
import pdb

from skimage.util import invert 

# building the max tree.
def build_max_tree(image, connectivity=2):
    """Builds the max tree from an image

    Component trees represent the hierarchical structure of the connected
    components resulting from sequential thresholding operations applied to an
    image. A connected component at one level is parent of a component at a
    higher level if the latter is included in the first. A max-tree is an
    efficient representation of a component tree. A connected component at
    one level is represented by one reference pixel at this level, which is
    parent to all other pixels at that level and the reference pixel at the
    level above. The max-tree is the basis for many morphological operators,
    namely connected operators. 

    Parameters
    ----------
    img: ndarray
        The input image for which the max-tree is to be calculated.
        This image can be of any type.
    connectivity: unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. It is 1 for
        4-connectivity and 2 for 8-connectivity. Default value is 2.

    Returns
    -------
    parent: ndarray, int64
        The value of each pixel is the index of its parent in the ravelled
        array

    tree_traverser: 1D array, int64
        The ordered pixel indices (referring to the ravelled array). The pixels
        are ordered such that every pixel is preceded by its parent (except for
        the root which has no parent).

    References
    ----------
    .. [1] Salembier, P., Oliveras, A., & Garrido, L. (1998). Antiextensive
           Connected Operators for Image and Sequence Processing.
           IEEE Transactions on Image Processing, 7(4), 555-570.
    .. [2] Berger, C., Geraud, T., Levillain, R., Widynski, N., Baillard, A.,
           Bertin, E. (2007). Effective Component Tree Computation with
           Application to Pattern Recognition in Astronomical Imaging.
           In International Conference on Image Processing (ICIP) (pp. 41-44).
    .. [3] Najman, L., & Couprie, M. (2006). Building the component tree in
           quasi-linear time. IEEE Transactions on Image Processing, 15(11),
           3531-3539.
    .. [4] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
           Component Tree Computation Algorithms. IEEE Transactions on Image
           Processing, 23(9), 3885-3895.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.max_tree import build_max_tree

    We create a small sample image (Figure 1 from [4]) and build the max-tree.

    >>> image = np.array([[15, 13, 16], [12, 12, 10], [16, 12, 14]])
    >>> P, S = build_max_tree(image, connectivity=2)

    """
    # User defined masks are not allowed, as there might be more than one
    # connected component in the mask (and therefore not a single tree that
    # represents the image). Mask here is an image that is 0 on the border
    # and 1 everywhere else.
    mask_shrink = np.ones([x-2 for x in image.shape], bool)
    mask = np.pad(mask_shrink, 1, mode='constant')

    neighbors, offset = _validate_connectivity(image.ndim, connectivity,
                                               offset=None)

    # initialization of the parent image
    parent = np.zeros(image.shape, dtype=np.int64)

    # flat_neighborhood contains a list of offsets allowing one to find the neighbors
    # in the ravelled image.
    flat_neighborhood = _compute_neighbors(image, neighbors, offset).astype(np.int32)

    # pixels need to be sorted according to their grey level.
    tree_traverser = np.argsort(image.ravel(),
                                kind='quicksort').astype(np.int64)

    # call of cython function.
    _max_tree._build_max_tree(image.ravel(), mask.ravel().astype(np.uint8),
                              flat_neighborhood, 
                              np.array(image.shape, dtype=np.int32),
                              parent.ravel(), tree_traverser)

    return parent, tree_traverser

def area_opening(image, area_threshold, connectivity=2):
    """Performs an area opening of the image.

    Area opening removes all bright structures of an image with
    a surface smaller than area_threshold.
    The output image is thus the largest image smaller than the input
    for which all local maxima have at least a surface of
    area_threshold pixels.

    Area openings are similar to morphological openings, but
    they do not use a fixed structuring element, but rather a deformable
    one, with surface = area_threshold. Consequently, the area_opening
    with area_threshold=1 is the identity.

    Technically, this operator is based on the max-tree representation of
    the image. 

    Parameters
    ----------
    img: ndarray
        The input image for which the area_opening is to be calculated.
        This image can be of any type.
    area_threshold: unsigned int
        The size parameter (number of pixels).
    connectivity: unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. It is 1 for
        4-connectivity and 2 for 8-connectivity. Default value is 1.

    Returns
    -------
    output: ndarray
        Output image of the same shape and type as img.

    See also
    --------
    skimage.morphology.max_tree.area_closing
    skimage.morphology.max_tree.diameter_opening
    skimage.morphology.max_tree.diameter_closing


    References
    ----------
    .. [1] Vincent L., Proc. "Grayscale area openings and closings,
           their efficient implementation and applications",
           EURASIP Workshop on Mathematical Morphology and its
           Applications to Signal Processing, Barcelona, Spain, pp.22-27,
           May 1993.
    .. [2] Soille, P., "Morphological Image Analysis: Principles and
           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.
    .. [3] Salembier, P., Oliveras, A., & Garrido, L. (1998). Antiextensive
           Connected Operators for Image and Sequence Processing.
           IEEE Transactions on Image Processing, 7(4), 555-570.
    .. [4] Najman, L., & Couprie, M. (2006). Building the component tree in
           quasi-linear time. IEEE Transactions on Image Processing, 15(11),
           3531-3539.
    .. [5] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
           Component Tree Computation Algorithms. IEEE Transactions on Image
           Processing, 23(9), 3885-3895.

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import max_tree

    We create an image (quadratic function with a maximum in the center and
    4 additional local maxima.

    >>> w = 12
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 20 - 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:3,1:5] = 40; f[2:4,9:11] = 60; f[9:11,2:4] = 80
    >>> f[9:10,9:11] = 100; f[10,10] = 100
    >>> f = f.astype(np.int)

    We can calculate the area opening:

    >>> open = attribute.area_opening(f, 8, connectivity=1)

    The peaks with a surface smaller than 8 are removed.
    """
    output = image.copy()

    P, S = build_max_tree(image, connectivity)

    area = _max_tree._compute_area(image.ravel(), P.ravel(), S)

    _max_tree._direct_filter(image.ravel(), output.ravel(), P.ravel(), S,
                             area, area_threshold)
    return output

def area_closing(image, area_threshold, connectivity=2):
    """Performs an area closing of the image.

    Area closing removes all dark structures of an image with
    a surface smaller than area_threshold.
    The output image is larger than or equal to the input image
    for every pixel and all local minima have at least a surface of
    area_threshold pixels.

    Area closings are similar to morphological closings, but
    they do not use a fixed structuring element, but rather a deformable
    one, with surface = area_threshold. 

    Technically, this operator is based on the max-tree representation of
    the image. 

    Parameters
    ----------
    img: ndarray
        The input image for which the area_closing is to be calculated.
        This image can be of any type.
    area_threshold: unsigned int
        The size parameter (number of pixels).
    connectivity: unsigned int, optional
        The neighborhood connectivity. The integer represents the maximum
        number of orthogonal steps to reach a neighbor. It is 1 for
        4-connectivity and 2 for 8-connectivity. Default value is 1.

    Returns
    -------
    output: ndarray
        Output image of the same shape and type as img.

    See also
    --------
    skimage.morphology.max_tree.area_opening
    skimage.morphology.max_tree.diameter_opening
    skimage.morphology.max_tree.diameter_closing


    References
    ----------
    .. [1] Vincent L., Proc. "Grayscale area openings and closings,
           their efficient implementation and applications",
           EURASIP Workshop on Mathematical Morphology and its
           Applications to Signal Processing, Barcelona, Spain, pp.22-27,
           May 1993.
    .. [2] Soille, P., "Morphological Image Analysis: Principles and
           Applications" (Chapter 6), 2nd edition (2003), ISBN 3540429883.
    .. [3] Salembier, P., Oliveras, A., & Garrido, L. (1998). Antiextensive
           Connected Operators for Image and Sequence Processing.
           IEEE Transactions on Image Processing, 7(4), 555-570.
    .. [4] Najman, L., & Couprie, M. (2006). Building the component tree in
           quasi-linear time. IEEE Transactions on Image Processing, 15(11),
           3531-3539.
    .. [5] Carlinet, E., & Geraud, T. (2014). A Comparative Review of
           Component Tree Computation Algorithms. IEEE Transactions on Image
           Processing, 23(9), 3885-3895.


    Examples
    --------
    >>> import numpy as np
    >>> from skimage.morphology import max_tree

    >>> import numpy as np
    >>> from skimage.morphology import attribute

    We create an image (quadratic function with a minimum in the center and
    4 additional local minima.

    >>> w = 12
    >>> x, y = np.mgrid[0:w,0:w]
    >>> f = 180 + 0.2*((x - w/2)**2 + (y-w/2)**2)
    >>> f[2:3,1:5] = 160; f[2:4,9:11] = 140; f[9:11,2:4] = 120
    >>> f[9:10,9:11] = 100; f[10,10] = 100
    >>> f = f.astype(np.int)

    We can calculate the area closing:

    >>> closed = attribute.area_closing(f, 8, connectivity=1)

    All small minima are removed, and the remaining minima have at least 
    a size of 8.
    """

    #max_val = image.max()
    #image_inv = max_val - image
    image_inv = invert(image)
    output = image_inv.copy()

    P, S = build_max_tree(image_inv, connectivity)

    area = _max_tree._compute_area(image_inv.ravel(), P.ravel(), S)

    _max_tree._direct_filter(image_inv.ravel(), output.ravel(), P.ravel(), S,
                             area, area_threshold)
    output = invert(output)
    return output


