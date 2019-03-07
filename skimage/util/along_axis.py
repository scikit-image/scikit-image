import numpy as np
from numpy.core.numeric import (asanyarray,
                                zeros)
from numpy.core.multiarray import normalize_axis_index
from numpy.core.fromnumeric import transpose
from numpy.lib.index_tricks import ndindex

### all arrs have to be of equal dims

def apply_along_axis(func1d, axis, tuple_arr, *args, **kwargs):
    # handle negative axes
    tuple_arr = tuple([asanyarray(arr) for arr in tuple_arr])
    nd = tuple_arr[0].ndim
    axis = normalize_axis_index(axis, nd)

    # arr, with the iteration axis at the end
    in_dims = list(range(nd))
    inarr_views = tuple([transpose(arr, in_dims[:axis] + in_dims[axis+1:] + [axis]) for arr in tuple_arr])

    # compute indices for the iteration axes, and append a trailing ellipsis to
    # prevent 0d arrays decaying to scalars, which fixes gh-8642
    inds = ndindex(inarr_views[0].shape[:-1])
    inds = (ind + (Ellipsis,) for ind in inds)

    # invoke the function on the first item
    try:
        ind0 = next(inds)
    except StopIteration:
        raise ValueError('Cannot apply_along_axis when any iteration dimensions are 0')
    res = asanyarray(func1d(tuple([inarr_view[ind0] for inarr_view in inarr_views]), *args, **kwargs))

    # build a buffer for storing evaluations of func1d.
    # remove the requested axis, and add the new ones on the end.
    # laid out so that each write is contiguous.
    # for a tuple index inds, buff[inds] = func1d(inarr_view[inds])
    buff = zeros(inarr_views[0].shape[:-1] + res.shape, res.dtype)

    # permutation of axes such that out = buff.transpose(buff_permute)
    buff_dims = list(range(buff.ndim))
    buff_permute = (
        buff_dims[0 : axis] +
        buff_dims[buff.ndim-res.ndim : buff.ndim] +
        buff_dims[axis : buff.ndim-res.ndim]
    )

    # matrices have a nasty __array_prepare__ and __array_wrap__
    buff = res.__array_prepare__(buff)

    # save the first result, then compute and save all remaining results
    buff[ind0] = res
    for ind in inds:
        buff[ind] = asanyarray(func1d(tuple([inarr_view[ind] for inarr_view in inarr_views]), *args, **kwargs))

    # wrap the array, to preserve subclasses
    buff = res.__array_wrap__(buff)

    # finally, rotate the inserted axes back to where they belong
    return transpose(buff, buff_permute)
