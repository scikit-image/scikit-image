import functools
import warnings

import numpy as np
from numpy import dtype, ndarray
from uarray import generate_multimethod, Dispatchable
from uarray import all_of_type, create_multimethod

from .._backend import _mark_output, _mark_scalar_or_array
from .filters import gaussian as _gaussian

__all__ = [
    "gaussian",
]

create_skimage_filters = functools.partial(
    create_multimethod, domain="numpy.skimage.filters"
)


def _image_arg_replacer(args, kwargs, dispatchables):
    """
    uarray argument replacer to replace the input image (``image``) and
    """

    def self_method(image, *args, **kwargs):
        return (dispatchables[0],) + args, kwargs

    return self_method(*args, **kwargs)


@create_skimage_filters(_image_arg_replacer)
@all_of_type(ndarray)
def gaussian(
    image,
    sigma=1,
    output=None,
    mode="nearest",
    cval=0,
    preserve_range=False,
    truncate=4.0,
    *,
    channel_axis=None
):
    return (image, _mark_output(output))


gaussian.__doc__ = _gaussian.__doc__
