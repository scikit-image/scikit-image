from skimage._shared.utils import skimage_deprecation
from warnings import warn
warn(skimage_deprecation('The `skimage.filter` module has been renamed to '
                         '`skimage.filters`.  This placeholder module '
                         'will be removed in v0.13.'))
del warn
del skimage_deprecation

from ..filters import *
