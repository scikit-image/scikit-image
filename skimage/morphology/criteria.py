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

from . import extrema
from .watershed import _validate_connectivity
from .watershed import _compute_neighbors

from .extrema import local_minima

def criteria_opening(image, crit_val, mask=None, 
                     connectivity=1):

    if mask is not None and mask.shape != image.shape:
        raise ValueError("mask must have same shape as image")
    if mask is None:
        # Use a complete `True` mask if none is provided
        mask = np.ones(image.shape, bool)

    connectivity, offset = _validate_connectivity(image.ndim, connectivity,
                                                  offset=None)

    # TODO : fix the structuring element issue. 
    seeds = local_minima(image)

    image = np.pad(image, 1, mode='constant')
    mask = np.pad(mask, pad_width, mode='constant').ravel()
    output = np.pad(seeds, 1, mode='constant')

    flat_neighborhood = _compute_neighbors(image, connectivity, offset)
    image_strides = np.array(image.strides, dtype=np.int32) // image.itemsize

