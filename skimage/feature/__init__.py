from .._shared.utils import deprecated

from ._canny import canny
from ._cascade import Cascade
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
                     structure_tensor_eigenvalues,
                     structure_tensor_eigvals, hessian_matrix,
                     hessian_matrix_eigvals, hessian_matrix_det,
                     corner_moravec, corner_orientations,
                     shape_index)
from .template import match_template
from .brief import BRIEF
from .censure import CENSURE
from .orb import ORB
from .match import match_descriptors
from .util import plot_matches
from .blob import blob_dog, blob_log, blob_doh
from .haar import (haar_like_feature, haar_like_feature_coord,
                   draw_haar_like_feature)
from ._basic_features import multiscale_basic_features


@deprecated(alt_func='skimage.registration.phase_cross_correlation',
            removed_version='0.19')
def masked_register_translation(src_image, target_image, src_mask,
                                target_mask=None, overlap_ratio=0.3):
    from ..registration import phase_cross_correlation
    return phase_cross_correlation(src_image, target_image,
                                   reference_mask=src_mask,
                                   moving_mask=target_mask,
                                   overlap_ratio=overlap_ratio)


@deprecated(alt_func='skimage.registration.phase_cross_correlation',
            removed_version='0.19')
def register_translation(src_image, target_image, upsample_factor=1,
                         space="real", return_error=True):
    from ..registration import phase_cross_correlation
    return phase_cross_correlation(src_image, target_image,
                                   upsample_factor=upsample_factor,
                                   space=space, return_error=return_error)


__all__ = ['canny',
           'Cascade',
           'daisy',
           'hog',
           'greycomatrix',
           'greycoprops',
           'local_binary_pattern',
           'multiblock_lbp',
           'draw_multiblock_lbp',
           'peak_local_max',
           'structure_tensor',
           'structure_tensor_eigenvalues',
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
           'masked_register_translation',
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
           'draw_haar_like_feature',
           'multiscale_basic_features',
           ]
