"""criteria.py - apply criteria based openings and closings

This module provides functions to apply criteria based openings and
closings to arbitrary images. These operators build on the flooding
algorithm used in the watershed transformation, but instead of
partitioning the image plane they stop the flooding once a certain
criterion is met by the region (often termed as 'lake'). The value
of the pixels belonging to this lake are set to the flooding level,
when the flooding stopped.

This implementation provides functions for
1. area openings/closings
2. volume openings/closings
3. diameter openings/closings

Dynamics openings and closings can be implemented by greyreconstruct.
They are therefore not implemented here.
"""
import numpy as np

from .watershed import _validate_connectivity
from .watershed import _compute_neighbors

from .extrema import local_minima
from ..measure import label

from . import _criteria

from ..util import crop
import pdb

# area_closing is by far the most popular of these operators.
def area_closing(image, area_threshold, mask=None, connectivity=1):

    if mask is not None and mask.shape != image.shape:
        raise ValueError("mask must have same shape as image")
    if mask is None:
        # Use a complete `True` mask if none is provided
        mask = np.ones(image.shape, bool)

    neighbors, offset = _validate_connectivity(image.ndim, connectivity,
                                               offset=None)

    seeds_bin = local_minima(image, selem = neighbors)
    seeds = label(seeds_bin, connectivity = connectivity).astype(np.uint64)
    output = image.copy()

    image = np.pad(image, 1, mode='constant')
    mask = np.pad(mask, 1, mode='constant')
    seeds = np.pad(seeds, 1, mode='constant')
    output = np.pad(output, 1, mode='constant')

    flat_neighborhood = _compute_neighbors(image, neighbors, offset)
    image_strides = np.array(image.strides, dtype=np.int32) // image.itemsize

    _criteria.area_closing(image.ravel(),
                           area_threshold,
                           seeds.ravel(),
                           flat_neighborhood,
                           mask.ravel().astype(np.uint8),
                           image_strides,
                           0.000001,
                           output.ravel()
                           )
    output = crop(output, 1, copy=True)

    return(output)

def volume_fill(image, volume_threshold, mask=None, connectivity=1):

    if mask is not None and mask.shape != image.shape:
        raise ValueError("mask must have same shape as image")
    if mask is None:
        # Use a complete `True` mask if none is provided
        mask = np.ones(image.shape, bool)

    neighbors, offset = _validate_connectivity(image.ndim, connectivity,
                                                  offset=None)

    seeds_bin = local_minima(image, selem = neighbors)
    seeds = label(seeds_bin, connectivity = connectivity).astype(np.uint64)
    output = image.copy()

    image = np.pad(image, 1, mode='constant')
    mask = np.pad(mask, 1, mode='constant')
    seeds = np.pad(seeds, 1, mode='constant')
    output = np.pad(output, 1, mode='constant')

    flat_neighborhood = _compute_neighbors(image, neighbors, offset)
    image_strides = np.array(image.strides, dtype=np.int32) // image.itemsize

    _criteria.volume_fill(image.ravel(),
                          volume_threshold,
                          seeds.ravel(),
                          flat_neighborhood,
                          mask.ravel().astype(np.uint8),
                          image_strides,
                          0.000001,
                          output.ravel()
                          )
    output = crop(output, 1, copy=True)

    return(output)


def diameter_closing(image, diameter_threshold, mask=None, connectivity=1):

    if mask is not None and mask.shape != image.shape:
        raise ValueError("mask must have same shape as image")
    if mask is None:
        # Use a complete `True` mask if none is provided
        mask = np.ones(image.shape, bool)

    neighbors, offset = _validate_connectivity(image.ndim, connectivity,
                                                  offset=None)

    seeds_bin = local_minima(image, selem = neighbors)
    seeds = label(seeds_bin, connectivity = connectivity).astype(np.uint64)
    output = image.copy()

    image = np.pad(image, 1, mode='constant')
    mask = np.pad(mask, 1, mode='constant')
    seeds = np.pad(seeds, 1, mode='constant')
    output = np.pad(output, 1, mode='constant')

    flat_neighborhood = _compute_neighbors(image, neighbors, offset)
    image_strides = np.array(image.strides, dtype=np.int32) // image.itemsize

    _criteria.diameter_closing(image.ravel(),
                               diameter_threshold,
                               seeds.ravel(),
                               flat_neighborhood,
                               mask.ravel().astype(np.uint8),
                               image_strides,
                               0.000001,
                               output.ravel()
                               )
    output = crop(output, 1, copy=True)

    return(output)
