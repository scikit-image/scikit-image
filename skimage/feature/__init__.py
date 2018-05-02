"""Feature detection subpackage.

In computer vision and image processing feature detection includes methods for
computing abstractions of image information and making local decisions at every
image point whether there is an image feature of a given type at that point or
not. The resulting features will be subsets of the image domain, often in the
form of isolated points, continuous curves or connected regions [1]_.

.. [1] https://en.wikipedia.org/wiki/Feature_detection_(computer_vision)

"""


from ._canny import canny
from ._daisy import daisy
from ._hog import hog
from .texture import (greycomatrix, greycoprops,
                      local_binary_pattern,
                      multiblock_lbp,
                      draw_multiblock_lbp)

from .peak import peak_local_max
from .corner import (corner_kitchen_rosenfeld, corner_harris,
                     corner_shi_tomasi, corner_foerstner, corner_subpix,
                     corner_peaks, corner_fast, structure_tensor,
                     structure_tensor_eigvals, hessian_matrix,
                     hessian_matrix_eigvals, hessian_matrix_det,
                     corner_moravec, corner_orientations,
                     shape_index)
from .template import match_template
from .register_translation import register_translation
from .brief import BRIEF
from .censure import CENSURE
from .orb import ORB
from .match import match_descriptors
from .util import plot_matches
from .blob import blob_dog, blob_log, blob_doh
from .haar import (haar_like_feature, haar_like_feature_coord,
                   draw_haar_like_feature)


__all__ = ['canny',
           'daisy',
           'hog',
           'greycomatrix',
           'greycoprops',
           'local_binary_pattern',
           'multiblock_lbp',
           'draw_multiblock_lbp',
           'peak_local_max',
           'structure_tensor',
           'structure_tensor_eigvals',
           'hessian_matrix',
           'hessian_matrix_det',
           'hessian_matrix_eigvals',
           'shape_index',
           'corner_kitchen_rosenfeld',
           'corner_harris',
           'corner_shi_tomasi',
           'corner_foerstner',
           'corner_subpix',
           'corner_peaks',
           'corner_moravec',
           'corner_fast',
           'corner_orientations',
           'match_template',
           'register_translation',
           'BRIEF',
           'CENSURE',
           'ORB',
           'match_descriptors',
           'plot_matches',
           'blob_dog',
           'blob_doh',
           'blob_log',
           'haar_like_feature',
           'haar_like_feature_coord',
           'draw_haar_like_feature']
