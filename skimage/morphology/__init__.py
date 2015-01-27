from .binary import (binary_erosion, binary_dilation, binary_opening,
                     binary_closing)
from .grey import (erosion, dilation, opening, closing, white_tophat,
                   black_tophat)
from .selem import (square, rectangle, diamond, disk, cube, octahedron, ball,
                    octagon, star)
from .watershed import watershed
from ._skeletonize import skeletonize, medial_axis
from .convex_hull import convex_hull_image, convex_hull_object
from .greyreconstruct import reconstruction
from .misc import remove_small_objects

from ..measure._label import label
from .._shared.utils import deprecated as _deprecated
label = _deprecated('skimage.measure.label')(label)


__all__ = ['binary_erosion',
           'binary_dilation',
           'binary_opening',
           'binary_closing',
           'erosion',
           'dilation',
           'opening',
           'closing',
           'white_tophat',
           'black_tophat',
           'square',
           'rectangle',
           'diamond',
           'disk',
           'cube',
           'octahedron',
           'ball',
           'octagon',
           'label',
           'watershed',
           'skeletonize',
           'medial_axis',
           'convex_hull_image',
           'convex_hull_object',
           'reconstruction',
           'remove_small_objects']
