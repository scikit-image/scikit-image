"""Algorithms to partition images into meaningful regions or boundaries."""

from ._expand_labels import expand_labels
from .random_walker_segmentation import random_walker
from .active_contour_model import active_contour
from ._felzenszwalb import felzenszwalb
from .slic_superpixels import slic
from ._quickshift import quickshift
from .boundaries import find_boundaries, mark_boundaries
from ._clear_border import clear_border
from ._join import join_segmentations, relabel_sequential
from ._watershed import watershed
from ._chan_vese import chan_vese
from .morphsnakes import (
    morphological_geodesic_active_contour,
    morphological_chan_vese,
    inverse_gaussian_gradient,
    disk_level_set,
    checkerboard_level_set,
)
from ..morphology import flood, flood_fill
from ._thresholding_global import (
    threshold_isodata,
    threshold_li,
    threshold_mean,
    threshold_minimum,
    threshold_multiotsu,
    threshold_otsu,
    threshold_triangle,
    threshold_yen,
    threshold_try_global,
)
from ._thresholding_local import (
    threshold_local,
    threshold_local_niblack,
    threshold_local_sauvola,
    threshold_labels_hysteresis,
)


__all__ = [
    'expand_labels',
    'random_walker',
    'active_contour',
    'felzenszwalb',
    'slic',
    'quickshift',
    'find_boundaries',
    'mark_boundaries',
    'clear_border',
    'join_segmentations',
    'relabel_sequential',
    'watershed',
    'chan_vese',
    'morphological_geodesic_active_contour',
    'morphological_chan_vese',
    'inverse_gaussian_gradient',
    'disk_level_set',
    'checkerboard_level_set',
    'flood',
    'flood_fill',
    'threshold_isodata',
    'threshold_li',
    'threshold_local',
    'threshold_mean',
    'threshold_minimum',
    'threshold_multiotsu',
    'threshold_otsu',
    'threshold_triangle',
    'threshold_yen',
    'threshold_try_global',
    'threshold_local_niblack',
    'threshold_local_sauvola',
    'threshold_labels_hysteresis',
]
