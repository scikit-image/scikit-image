from _skimage2.transform._warps import (
    downscale_local_mean as downscale_local_mean,
    rescale as rescale,
    resize as resize,
    resize_local_mean as resize_local_mean,
    rotate as rotate,
    swirl as swirl,
    warp as warp,
    warp_coords as warp_coords,
    warp_polar as warp_polar,
)  # noqa: F401

from skimage._doctest_adapters import adapt_doctests

# We need the shim versions of these classes.
from ._geometric import SimilarityTransform, AffineTransform, ProjectiveTransform

from _skimage2.transform._warps import (  # noqa: F401
    _linear_polar_mapping,
    _log_polar_mapping,
    _stackcopy,
)

HOMOGRAPHY_TRANSFORMS = (SimilarityTransform, AffineTransform, ProjectiveTransform)

__all__ = [
    'HOMOGRAPHY_TRANSFORMS',
    'downscale_local_mean',
    'rescale',
    'resize',
    'resize_local_mean',
    'rotate',
    'swirl',
    'warp',
    'warp_coords',
    'warp_polar',
]

adapt_doctests(globals())
