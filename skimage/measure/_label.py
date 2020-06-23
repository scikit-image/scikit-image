""" 
expand_labels is  inspired by code in CellProfiler, 
code licensed under BSD-3 licenses.
Website: http://www.cellprofiler.org

Copyright (c) 2020 Broad Institute
All rights reserved.

Original author/s: Cellprofiler team 
"""

from ._ccomp import label_cython as clabel
from scipy.ndimage import distance_transform_edt
import numpy as np

def label(input, neighbors=None, background=None, return_num=False,
          connectivity=None):
    r"""Label connected regions of an integer array.

    Two pixels are connected when they are neighbors and have the same value.
    In 2D, they can be neighbors either in a 1- or 2-connected sense.
    The value refers to the maximum number of orthogonal hops to consider a
    pixel/voxel a neighbor::

      1-connectivity     2-connectivity     diagonal connection close-up

           [ ]           [ ]  [ ]  [ ]             [ ]
            |               \  |  /                 |  <- hop 2
      [ ]--[x]--[ ]      [ ]--[x]--[ ]        [x]--[ ]
            |               /  |  \             hop 1
           [ ]           [ ]  [ ]  [ ]

    Parameters
    ----------
    input : ndarray of dtype int
        Image to label.
    neighbors : {4, 8}, int, optional
        Whether to use 4- or 8-"connectivity".
        In 3D, 4-"connectivity" means connected pixels have to share face,
        whereas with 8-"connectivity", they have to share only edge or vertex.
        **Deprecated, use** ``connectivity`` **instead.**
    background : int, optional
        Consider all pixels with this value as background pixels, and label
        them as 0. By default, 0-valued pixels are considered as background
        pixels.
    return_num : bool, optional
        Whether to return the number of assigned labels.
    connectivity : int, optional
        Maximum number of orthogonal hops to consider a pixel/voxel
        as a neighbor.
        Accepted values are ranging from  1 to input.ndim. If ``None``, a full
        connectivity of ``input.ndim`` is used.

    Returns
    -------
    labels : ndarray of dtype int
        Labeled array, where all connected regions are assigned the
        same integer value.
    num : int, optional
        Number of labels, which equals the maximum label index and is only
        returned if return_num is `True`.

    See Also
    --------
    regionprops

    References
    ----------
    .. [1] Christophe Fiorio and Jens Gustedt, "Two linear time Union-Find
           strategies for image processing", Theoretical Computer Science
           154 (1996), pp. 165-181.
    .. [2] Kensheng Wu, Ekow Otoo and Arie Shoshani, "Optimizing connected
           component labeling algorithms", Paper LBNL-56864, 2005,
           Lawrence Berkeley National Laboratory (University of California),
           http://repositories.cdlib.org/lbnl/LBNL-56864

    Examples
    --------
    >>> import numpy as np
    >>> x = np.eye(3).astype(int)
    >>> print(x)
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    >>> print(label(x, connectivity=1))
    [[1 0 0]
     [0 2 0]
     [0 0 3]]
    >>> print(label(x, connectivity=2))
    [[1 0 0]
     [0 1 0]
     [0 0 1]]
    >>> print(label(x, background=-1))
    [[1 2 2]
     [2 1 2]
     [2 2 1]]
    >>> x = np.array([[1, 0, 0],
    ...               [1, 1, 5],
    ...               [0, 0, 0]])
    >>> print(label(x))
    [[1 0 0]
     [1 1 2]
     [0 0 0]]
    """
    return clabel(input, neighbors, background, return_num, connectivity)


def expand_labels(label_image, distance):
    """Expand labels in label image by ``distance`` pixels without overlapping.

    Given a label image, each label is grown by up to distance pixels. 
    However, where labels would start to overlap, the label growth may
    stop at less than distance pixels (this is where it differs from a 
    morphological dilation, where a connected component with a high label 
    number can potentially override connected components with lower label
    numbers).

    This is equivalent to CellProfiler [1] [2] IdentifySecondaryObjects method
    using the option "Distance-N".

    The basic idea is that you have some seed labels that you want 
    to grow by n pixels to give a mask for a larger object.
    
    If you were only dealing with a single seed object, you could simply
    dilate with a suitably sized structuring element. However, in general you
    have multiple seed points and you don't want to merge those. Distance N
    will grow up to N pixels without merging objects that are closer together
    than 2N.
    
    There is an important edge case when a pixel has the same distance to
    multiple regions, as it is not defined which region expands into that
    space, see the discussion in [1]. Here, the exact bahaviour depends on
    the upstream implementation.

    Parameters
    ----------
    label_image : ndarray of dtype int
        label image
    distance : float
        Number of pixels by which to grow the labels.

    Returns
    -------
    enlarged_labels : ndarray of dtype int
        Labeled array, where all connected regions have been enlarged

    See Also
    --------
    :py:func:`label`, :py:func:`skimage.segmentation.watershed`

    References
    ----------
    .. [1] https://cellprofiler.org
    .. [2] https://github.com/CellProfiler/CellProfiler/blob/082930ea95add7b72243a4fa3d39ae5145995e9c/cellprofiler/modules/identifysecondaryobjects.py#L559
    .. [3] https://forum.image.sc/t/equivalent-to-cellprofilers-identifysecondaryobjects-distance-n-in-fiji/39146/16

    Examples
    --------
    >>> labels = np.array([0, 1, 0, 0, 0, 0, 2])
    >>> expand_labels(labels, distance=1)
    array([1, 1, 1, 0, 0, 2, 2])

    Labels will not overwrite each other:

    >>> expand_labels(labels, distance=3)
    array([1, 1, 1, 1, 2, 2, 2])

    In case of ties, behavior is undefined, but currently resolves to the
    label closest to ``(0,) * ndim`` in lexicographical order.

    >>> labels_tied = np.array([0, 1, 0, 2, 0])
    >>> expand_labels(labels_tied, 1)
    array([1, 1, 1, 2, 2])
    >>> labels2d = np.array(
    ...     [[0, 1, 0, 0],
    ...      [2, 0, 0, 0],
    ...      [0, 3, 0, 0]]
    ... )
    >>> expand_labels(labels2d, 1)
    array([[2, 1, 1, 0],
           [2, 2, 0, 0],
           [2, 3, 3, 0]])
    """
    
    distances, nearest_label_coords = distance_transform_edt(
        label_image == 0, return_indices=True
    )
    labels_out = np.zeros(label_image.shape, label_image.dtype)
    dilate_mask = distances <= distance
    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out
