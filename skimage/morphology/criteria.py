"""extrema.py - local minima and maxima

This module provides functions to apply criteria based openings and
closings. These operators build on the flooding algorithm used in the
watershed transformation, but instead of partitioning the image plane
they stop the flooding once a certain criterion is met by the region
(often termed as 'lake'). The value of the pixels belonging to this
lake are set to the flooding level, when the flooding stopped.

This implementation provides functions for
1. dynamics openings/closings
2. area openings/closings
3. volume openings/closings
4. diameter openings/closings

Dynamics openings and closings can also be implemented by greyreconstruct.
"""
import numpy as np

from .watershed import _validate_connectivity
from .watershed import _compute_neighbors

from .extrema import local_minima
from ..measure import label

from . import _criteria

from ..util import crop

import pdb


def area_closing(image, area_threshold, mask=None, connectivity=1,
                 compactness=0.0):

    if mask is not None and mask.shape != image.shape:
        raise ValueError("mask must have same shape as image")
    if mask is None:
        # Use a complete `True` mask if none is provided
        mask = np.ones(image.shape, bool)

    connectivity, offset = _validate_connectivity(image.ndim, connectivity,
                                                  offset=None)

    # TODO : fix the structuring element issue.
    # labeling and minima detection need to rely on the same connectivity.
    seeds_bin = local_minima(image)
    seeds = label(seeds_bin).astype(np.uint64)
    output = image.copy()

    image = np.pad(image, 1, mode='constant')
    mask = np.pad(mask, 1, mode='constant')
    seeds = np.pad(seeds, 1, mode='constant')
    output = np.pad(output, 1, mode='constant')

    flat_neighborhood = _compute_neighbors(image, connectivity, offset)
    image_strides = np.array(image.strides, dtype=np.int32) // image.itemsize

    _criteria.area_closing(image.ravel(),
                           area_threshold,
                           seeds.ravel(),
                           flat_neighborhood,
                           mask.ravel().astype(np.uint8),
                           image_strides,
                           0.000001,
                           compactness,
                           output.ravel()
                           )
    output = crop(output, 1, copy=True)

    return(output)


