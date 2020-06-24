import numpy as np
from scipy.ndimage import distance_transform_edt


def expand_labels(label_image, distance):
    """Expand labels in label image by ``distance`` pixels without overlapping.

    Given a label image, `expand_labels` expands each connected component by up
    to ``distance`` pixels by assigning the integer label of a connected component 
    to background pixels that are within a Euclidean distance of <= ``distance`` 
    pixels of that component. Where multiple connected components are within 
    ``distance`` pixels of a background pixel, the label value of the closest
    connected component will be assigned (see Notes for multiple labels at equal 
    distance).
     

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

    Notes
    -----
    Where labels are spaced more than ``distance`` pixels are apart, this is
    equivalent to a morpholgical dilation with a disc or hyperball of radius ``distance``.
    However, in contrast to a morphological dilation, ``expand_labels`` will
    not expand a region into a neighbouring region.  

    This implementation of ``expand_labels`` is derived from CellProfiler [1], where
    it is known as module "IdentifySecondaryObjects (Distance-N)" [2].

    There is an important edge case when a pixel has the same distance to
    multiple regions, as it is not defined which region expands into that
    space. Here, the exact bahaviour depends on the upstream implementation.

    See Also
    --------
    :func:`skimage.measure.label`, :func:`skimage.segmentation.watershed`, :func:`skimage.morphology.dilation`

    References
    ----------
    .. [1] https://cellprofiler.org
    .. [2] https://github.com/CellProfiler/CellProfiler/blob/082930ea95add7b72243a4fa3d39ae5145995e9c/cellprofiler/modules/identifysecondaryobjects.py#L559

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
    labels_out = np.zeros_like(label_image)
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
