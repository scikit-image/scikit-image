from .binary import binary_closing, binary_dilation, binary_erosion, binary_opening

from .isotropic import (
    isotropic_erosion,
    isotropic_dilation,
    isotropic_opening,
    isotropic_closing,
)

from .footprints import (
    ball,
    cube,
    diamond,
    disk,
    ellipse,
    footprint_from_sequence,
    footprint_rectangle,
    octagon,
    octahedron,
    rectangle,
    square,
    star,
)

from ._footprints import mirror_footprint, pad_footprint

from ._grayscale_operators import (
    erosion,
    dilation,
    opening,
    closing,
    white_tophat,
    black_tophat,
)

from ._skeletonize import medial_axis, skeletonize, thin
from .convex_hull import convex_hull_image, convex_hull_object
from .grayreconstruct import reconstruction
from .misc import remove_small_holes, remove_small_objects, remove_objects_by_distance
from .extrema import h_maxima, h_minima, local_minima, local_maxima
from ._flood_fill import flood, flood_fill
from ._max_tree import (
    area_opening,
    area_closing,
    diameter_closing,
    diameter_opening,
    max_tree,
    max_tree_local_maxima,
)

def label(label_image, background=None, return_num=False, connectivity=None): ...

__all__ = [
    'area_closing',
    'area_opening',
    'ball',
    'black_tophat',
    'closing',
    'convex_hull_image',
    'convex_hull_object',
    'diameter_closing',
    'diameter_opening',
    'diamond',
    'dilation',
    'disk',
    'ellipse',
    'erosion',
    'flood',
    'flood_fill',
    'footprint_from_sequence',
    'footprint_rectangle',
    'h_maxima',
    'h_minima',
    'isotropic_closing',
    'isotropic_dilation',
    'isotropic_erosion',
    'isotropic_opening',
    'label',
    'local_maxima',
    'local_minima',
    'max_tree',
    'max_tree_local_maxima',
    'medial_axis',
    'mirror_footprint',
    'octagon',
    'octahedron',
    'opening',
    'pad_footprint',
    'reconstruction',
    'remove_small_holes',
    'remove_small_objects',
    'remove_objects_by_distance',
    'skeletonize',
    'star',
    'thin',
    'white_tophat',
]
