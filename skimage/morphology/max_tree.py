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
4. ellipse fit filter

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
    level above.

    Parameters
    ----------
    img: ndarray
        The input image for which the area_closing is to be calculated.
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

    parent = np.zeros(image.shape, dtype=np.int64)

    flat_neighborhood = _compute_neighbors(image, neighbors, offset).astype(np.int32)
    image_strides = np.array(image.strides, dtype=np.int32) // image.itemsize

    # pixels need to be sorted according to their grey level.
    tree_traverser = np.argsort(image.ravel(),
                                kind='quicksort').astype(np.int64)

    # call of cython function.
    _max_tree._build_max_tree(image.ravel(), mask.ravel().astype(np.uint8),
                              flat_neighborhood, image_strides,
                              np.array(image.shape, dtype=np.int32),
                              parent.ravel(), tree_traverser)

    return parent, tree_traverser

def area_open(image, area_threshold, connectivity=2):
    output = image.copy()

    P, S = build_max_tree(image, connectivity)

    area = _max_tree._compute_area(image.ravel(), P.ravel(), S)

    _max_tree._direct_filter(image.ravel(), output.ravel(), P.ravel(), S,
                             area, area_threshold)
    return output

def area_close(image, area_threshold, connectivity=2):
    max_val = image.max()
    image_inv = max_val - image
    output = image_inv.copy()

    P, S = build_max_tree(image_inv, connectivity)

    area = _max_tree._compute_area(image_inv.ravel(), P.ravel(), S)

    _max_tree._direct_filter(image_inv.ravel(), output.ravel(), P.ravel(), S,
                             area, area_threshold)
    output = max_val - output
    return output

def ellipse_filter(image, ratio_threshold, connectivity=2,
                   area_low_thresh=0, area_high_thresh=None,
                   method='direct'):
    
    if area_high_thresh is None:
        area_high_thresh = image.shape[0] * image.shape[1]
        
    output = image.copy()

    P, S = build_max_tree(image, connectivity)

    image_strides = np.array(image.strides, dtype=np.int32) // image.itemsize
    if method == 'cut_first':
        ellipse_area_ratio = _max_tree._compute_ellipse_ratio_2d(image.ravel(),
                                                                 P.ravel(), 
                                                                 S,
                                                                 image_strides,
                                                                 area_low_thresh,
                                                                 area_high_thresh)
        ellipse_area_ratio[ellipse_area_ratio > 1.0] = 1.0
        ellipse_area_ratio[ellipse_area_ratio < 0.0] = 0.0    
        _max_tree._cut_first_filter(image.ravel(), output.ravel(), P.ravel(), S,
                                    ellipse_area_ratio, ratio_threshold)
    elif method == 'direct':
        ellipse_area_ratio = _max_tree._compute_ellipse_ratio_2d(image.ravel(),
                                                                 P.ravel(), 
                                                                 S,
                                                                 image_strides,
                                                                 area_low_thresh,
                                                                 area_high_thresh)
        
        ellipse_area_ratio[ellipse_area_ratio > 1.0] = 1.0
        ellipse_area_ratio[ellipse_area_ratio < 0.0] = 0.0
        ellipse_area_ratio = 1.0 - ellipse_area_ratio
        _max_tree._direct_filter(image.ravel(), output.ravel(), P.ravel(), S,
                                    ellipse_area_ratio, ratio_threshold)
    else:
        raise ValueError('Method is not implemented. Please choose among direct and cut_first.')
    
    return output


def old_ellipse_filter(image, ratio_threshold, connectivity=2):
    output = image.copy()

    P, S = build_max_tree(image, connectivity)

    image_strides = np.array(image.strides, dtype=np.int32) // image.itemsize
    ellipse_area_ratio = _max_tree._compute_ellipse_ratio_2d(image.ravel(),
                                                             P.ravel(), S,
                                                             image_strides)

    _max_tree._direct_filter(image.ravel(), output.ravel(), P.ravel(), S,
                             ellipse_area_ratio, ratio_threshold)
    return output

