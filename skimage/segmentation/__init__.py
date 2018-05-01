"""Subpackage for image segmentation.

Image segmentation is the process of partitioning a digital image into multiple
segments. The goal of segmentation is to simplify and/or change the
representation of an image into something that is more meaningful and easier to
analyze [1]_. The subpackage contains methods, e.g., inspired from graph theory
and elastic energy minimization.

.. [1] https://en.wikipedia.org/wiki/Image_segmentation

"""


from .random_walker_segmentation import random_walker
from .active_contour_model import active_contour
from ._felzenszwalb import felzenszwalb
from .slic_superpixels import slic
from ._quickshift import quickshift
from .boundaries import find_boundaries, mark_boundaries
from ._clear_border import clear_border
from ._join import join_segmentations, relabel_from_one, relabel_sequential
from ..morphology import watershed
from ._chan_vese import chan_vese
from .morphsnakes import (morphological_geodesic_active_contour,
                          morphological_chan_vese, inverse_gaussian_gradient,
                          circle_level_set, checkerboard_level_set)


__all__ = ['random_walker',
           'active_contour',
           'felzenszwalb',
           'slic',
           'quickshift',
           'find_boundaries',
           'mark_boundaries',
           'clear_border',
           'join_segmentations',
           'relabel_from_one',
           'relabel_sequential',
           'watershed',
           'chan_vese',
           'morphological_geodesic_active_contour',
           'morphological_chan_vese',
           'inverse_gaussian_gradient',
           'circle_level_set',
           'checkerboard_level_set'
           ]
