from skimage._shared.utils import skimage_deprecation
from warnings import warn_explicit
__warningregistry__ = {}
warn_explicit(skimage_deprecation('The `skimage.filter` module has been renamed to '
                                  '`skimage.filters`.  This placeholder module '
                                  'will be removed in v0.13.'), category=None,
                                  filename=__file__, lineno=4,
                                  registry=__warningregistry__)
del warn_explicit
del skimage_deprecation

from ..filters import *
