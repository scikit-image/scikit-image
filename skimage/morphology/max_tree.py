"""max_tree.py - max-tree representation of an image and related operators
"""
import numpy as np

from .watershed import _validate_connectivity
from .watershed import _compute_neighbors

from . import _max_tree

from ..util import crop

def build_max_tree(image, connectivity=2):
    mask_shrink = np.ones([x-2 for x in image.shape], bool)
    mask = np.pad(mask_shrink, 1, mode='constant')

    neighbors, offset = _validate_connectivity(image.ndim, connectivity,
                                               offset=None)

    parent = np.zeros(image.shape, dtype=np.int64)

    flat_neighborhood = _compute_neighbors(image, neighbors, offset)
    image_strides = np.array(image.strides, dtype=np.int32) // image.itemsize

    tree_traverser = np.argsort(image.ravel(),
                                kind='quicksort').astype(np.int64)

    _max_tree._build_max_tree(image.ravel(), mask.ravel().astype(np.uint8),
                              flat_neighborhood, image_strides,
                              np.array(image.shape, dtype=np.int32),
                              parent.ravel(), tree_traverser)

    return parent, tree_traverser

def area_open(image, area_threshold, connectivity=2):
    output = image.copy()

    P, S = build_max_tree(image, connectivity)

    area = _max_tree._compute_area(image.ravel(), P.ravel(), S)

    _max_tree._apply_attribute(image.ravel(), output.ravel(), P.ravel(), S,
                               area, area_threshold)
    return output

def area_close(image, area_threshold, connectivity=2):
    max_val = image.max()
    image_inv = max_val - image
    output = image_inv.copy()

    P, S = build_max_tree(image_inv, connectivity)

    area = _max_tree._compute_area(image_inv.ravel(), P.ravel(), S)

    _max_tree._apply_attribute(image_inv.ravel(), output.ravel(), P.ravel(), S,
                               area, area_threshold)
    output = max_val - output
    return output
