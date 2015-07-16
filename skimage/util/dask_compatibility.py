from __future__ import division
import numpy as np
import dask.array as da
import operator
import collections
import functools
import six
import psutil      # New dependency

"""
Provides compatibility to use Dask and NumPy arrays on a shared codebase.

The goal of this module is to provide transparent compatibility functions
with the same names, signatures, and behavior from the base NumPy (``np``)
and Dask (``da``) namespaces. Ideally this module would be imported and used
as follows::

>>> import dask_compatibility as dc
>>> dc.rint(arr)  # Identical behavior if arr is from Dask or NumPy
"""

__all__ = ['asanyarray',
           'asarray',
           'clip',
           'dtype_conversion_inplace',
           'eager_min',
           'eager_max',
           'empty',
           'est_best_chunk',
           'floor',
           'maximum',
           'rint',
           ]


def _graceful_compute(variable):
    """
    Attempts to call .compute() on every item in variable.
    """
    # Shortcut on NumPy arrays first
    if isinstance(variable, np.ndarray):
        # Return the NumPy array
        return variable

    # Dask arrays are not iterable, but NumPy arrays are.
    elif not isinstance(variable, collections.Iterable):
        # Could be a Dask array, try to compute it
        try:
            return variable.compute()
        except AttributeError:
            # Not a Dask array, return the variable.
            return variable

    # Might be multiple values in a container
    else:
        # TODO: graceful computing of items in dicts
        try:
            # Attempt to gracefully compute each item
            return tuple(_graceful_compute(x) for x in variable)
        except TypeError:
            # If neither of these work it isn't a Dask array; return as-is
            return variable


def _dask_input(*args, **kwargs):
    """
    Determines if any inputs contain Dask arrays.

    Parameters
    ----------
    *args, **kwargs
        All arguments from the enclosing function should be passed.

    Returns
    -------
    dask_present : bool
        True if at least one input was a Dask array.
    """
    return (any(isinstance(x, da.core.Array) for x in args) or
            any(isinstance(x, da.core.Array)
                for _, x in six.iteritems(kwargs)))


def dask_decorator(compute=True):
    """
    Decorator to control conversion of outputs from Dask to NumPy arrays

    Parameters
    ----------
    compute : bool
        Decorator argument allows the default behavior (True, do the
        conversion) to be overridden so Dask arrays can be passed along
        to another function.

    Returns
    -------
    decorator : function
        Decorated function, ready for use.
    """
    def real_decorator(function):
        def wrapper(*args, **kwargs):
            if _dask_input(*args, **kwargs) or not compute:
                return function(*args, **kwargs)
            else:
                return _graceful_compute(function(*args, **kwargs))
        return wrapper
    return real_decorator


def _total_elements(shape):
    """
    Efficiently determine the number of elements in an array, given its shape.
    """
    return functools.reduce(operator.mul, shape, 1)


def est_best_chunk(shape, dtype, mem_frac=0.25, preserve_dims_under=50):
    """
    Find optimal dask.array chunks given shape and dtype of an array.

    Parameters
    ----------
    shape : iterable of ints
        Formatted like the ``numpy_array.shape`` attribute.
    dtype : NumPy dtype
        Actual NumPy dtype, e.g., ``np.float64`` or ``np.uint8``.
    frac_of_phys_mem : float
        Floating point value on interval (0, 1]. The chunk size found
        will fit in your total memory multiplied by this value.
    preserve_dims_under : int
        Integer value, must be >= 1. Dimensions with lengths below this value
        will not be broken down further.

    Returns
    -------
    smaller_chunk : iterable of ints
        Same length as ``shape``, this corresponds to an appropriate value for
        ``dask.array.from_array(..., chunks=smaller_chunk)``.
    """
    ram = psutil.virtual_memory().total * mem_frac

    # Size of each array element
    bytes_per_entry = np.dtype(dtype).itemsize

    # Start with current shape, decrease iteratively as necessary
    smaller_chunk = shape
    preserve_dims_under *= 2

    while (bytes_per_entry * _total_elements(smaller_chunk)) > ram:
        smaller_chunk = tuple(dim // 2
                              if dim > preserve_dims_under
                              else dim
                              for dim in smaller_chunk)

    return smaller_chunk


def asanyarray(arr_like):
    """
    Convert array-like input to array, passing dask arrays through.

    Parameters
    ----------
    arr_like : array-like
        Array-like input object to be converted to an array. Can be a NumPy
        array, a ``dask.array``, or any other array-like object.

    Returns
    -------
    arr : array
        If ``arr_like`` was a ``dask.array`` object it is passed unmodified,
        otherwise a NumPy array is returned.

    Notes
    -----
    This function is intended to replace ``np.asanyarray`` in the package so
    Dask arrays can be passed without conversion back into NumPy objects.
    """
    if isinstance(arr_like, da.core.Array):
        return arr_like
    else:
        return np.asanyarray(arr_like)


def asarray(arr_like):
    """
    Convert array-like input to array, passing dask arrays through.

    Parameters
    ----------
    arr_like : array-like
        Array-like input object to be converted to an array. Can be a NumPy
        array, a ``dask.array``, or any other array-like object.

    Returns
    -------
    arr : array
        If ``arr_like`` was a ``dask.array`` object it is passed unmodified,
        otherwise a NumPy array is returned.

    Notes
    -----
    This function is intended to replace ``np.asarray`` in the package so
    Dask arrays can be passed without conversion back into NumPy objects.
    """
    if isinstance(arr_like, da.core.Array):
        return arr_like
    else:
        return np.asarray(arr_like)


def dtype_conversion_inplace(arr, new_dtype):
    """
    Convert dtype of NumPy or dask array inplace.

    Parameters
    ----------
    arr : dask or NumPy array
        Input array, NumPy or Dask arrays supported.
    new_dtype : NumPy dtype object
        Valid destination dtype for ``arr``.

    Returns
    -------
    arr : dask or NumPy array
        Output array, converted to dtype ``new_dtype``.

    Notes
    -----
    This function is intended to replace ``np.array(..., copy=False)`` as
    Dask has no equivalent functionality.
    """
    # There is no `array` method in dask, so the code must branch
    if isinstance(arr, np.ndarray):
        return np.array(arr, new_dtype, copy=False)
    else:
        return arr.astype(new_dtype)


def eager_min(arr, *args, **kwargs):
    """
    Calculate the minimum of a NumPy or Dask array.

    Parameters
    ----------
    arr : Dask or NumPy array
        Input array, NumPy or Dask arrays supported.
    axis : int, optional
        Axis along which to operate. By default, flattened input is used.
    out : ndarray, optional
        Alternative output array in which to place the result. Must
        be of the same shape and buffer length as the expected output.
        See ``doc.ufuncs`` (Section "Output arguments") for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original ``arr``.

    Returns
    -------
    amax : ndarray or scalar
        Maximum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.
    """
    amin = arr.min(*args, **kwargs)

    if isinstance(arr, da.core.Array):
        amin = amin.compute()

    return amin


def eager_max(arr, *args, **kwargs):
    """
    Calculate the maximum of a NumPy or Dask array.

    Parameters
    ----------
    arr : Dask or NumPy array
        Input array, NumPy or Dask arrays supported.
    axis : int, optional
        Axis along which to operate. By default, flattened input is used.
    out : ndarray, optional
        Alternative output array in which to place the result. Must
        be of the same shape and buffer length as the expected output.
        See ``doc.ufuncs`` (Section "Output arguments") for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original ``arr``.

    Returns
    -------
    amax : ndarray or scalar
        Maximum of `a`. If `axis` is None, the result is a scalar value.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.

    Notes
    -----
    Use this function instead of ``np.max`` or ``arr.max()`` when the result
    must be known immediately within a function for other purposes. The
    result is calculated and returned from Dask.

    In all other cases simply use the ``arr.max()`` attribute which is shared
    between NumPy and Dask (but will remain lazy in Dask).
    """
    amin = arr.min(*args, **kwargs)

    if isinstance(arr, da.core.Array):
        amin = amin.compute()

    return amin


def rint(arr, out=None):
    """
    Calculate the maximum of a NumPy or Dask array.

    Parameters
    ----------
    arr : Dask or NumPy array
        Input array, NumPy or Dask arrays supported.
    out : ndarray, optional
        Alternative output array in which to place the result. Must
        be of the same shape and buffer length as the expected output.
        See ``doc.ufuncs`` (Section "Output arguments") for more details.

    Returns
    -------
    arr_rounded : Dask or NumPy array
        Rounded output array.
    """
    if isinstance(arr, da.core.Array):
        return da.rint(arr)
    else:
        return np.rint(arr, out=out)


def clip(arr, a_min, a_max, out=None):
    """
    Clip (limit) the values in an array.

    Given an interval, values outside the interval are clipped to
    the interval edges.  For example, if an interval of ``[0, 1]``
    is specified, values smaller than 0 become 0, and values larger
    than 1 become 1.

    Parameters
    ----------
    a : array_like
        Array containing elements to clip.
    a_min : scalar or array_like
        Minimum value.
    a_max : scalar or array_like
        Maximum value.  If ``a_min`` or ``a_max`` are array_like, then they
        will be broadcasted to the shape of ``a``.
    out : ndarray, optional
        The results will be placed in this array. It may be the input
        array for in-place clipping.  ``out`` must be of the right shape
        to hold the output.  Its type is preserved.

    Returns
    -------
    clipped_array : ndarray
        An array with the elements of ``a``, but where values
        < ``a_min`` are replaced with ``a_min``, and those > ``a_max``
        with ``a_max``.
    """
    if isinstance(arr, da.core.Array):
        return da.clip(arr, a_min, a_max)
    else:
        return np.clip(arr, a_min, a_max, out=out)


def floor(arr, out=None):
    """
    Return the floor of the input, element-wise.

    The floor of the scalar ``x`` is the largest integer ``i``, such that
    ``i <= x``.  It is often denoted as :math:`\lfloor x \rfloor`.

    Parameters
    ----------
    arr : array_like
        Input data.

    Returns
    -------
    arr_floored : {ndarray, scalar}
        The floor of each element in ``x``.
    """
    if isinstance(arr, da.core.Array):
        return da.floor(arr)
    else:
        return np.floor(arr, out=out)


def maximum(x1, x2, out=None, dtype=None, casting='safe'):
    """
    Element-wise maximum of array elements.

    Compare two arrays and returns a new array containing the element-wise
    maxima. If one of the elements being compared is a NaN, then that
    element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which
    are defined as at least one of the real or imaginary parts being a NaN.
    The net effect is that NaNs are propagated.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays holding the elements to be compared. They must have
        the same shape, or shapes that can be broadcast to a single shape.

    Returns
    -------
    y : {ndarray, scalar}
        The maximum of `x1` and `x2`, element-wise.  Returns scalar if
        both  `x1` and `x2` are scalars.
    """
    if isinstance(x1, da.core.Array) and isinstance(x2, da.core.Array):
        return da.maximum(x1, x2, dtype=dtype, casting=casting)
    else:
        if out is not None:
            if isinstance(out, da.core.Array):
                out = out.compute()
            return np.maximum(x1, x2, out=out, dtype=dtype, casting=casting)
        else:
            return np.maximum(x1, x2, dtype=dtype, casting=casting)


def minimum(x1, x2, out=None, dtype=None, casting='safe'):
    """
    Element-wise minimum of array elements.

    Compare two arrays and returns a new array containing the element-wise
    minima. If one of the elements being compared is a NaN, then that
    element is returned. If both elements are NaNs then the first is
    returned. The latter distinction is important for complex NaNs, which
    are defined as at least one of the real or imaginary parts being a NaN.
    The net effect is that NaNs are propagated.

    Parameters
    ----------
    x1, x2 : array_like
        The arrays holding the elements to be compared. They must have
        the same shape, or shapes that can be broadcast to a single shape.

    Returns
    -------
    y : {ndarray, scalar}
        The minimum of `x1` and `x2`, element-wise.  Returns scalar if
        both  `x1` and `x2` are scalars.
    """
    if isinstance(x1, da.core.Array) and isinstance(x2, da.core.Array):
        return da.minimum(x1, x2, dtype=dtype, casting=casting)
    else:
        if out is not None:
            if isinstance(out, da.core.Array):
                out = out.compute()
            return np.minimum(x1, x2, out=out, dtype=dtype, casting=casting)
        else:
            return np.minimum(x1, x2, dtype=dtype, casting=casting)


def empty(shape, dtype=np.float64, order='C', chunks=None):
    """
    Return a new array of given shape and type, without initializing entries.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the empty array
    dtype : data-type, optional
        Desired output data-type.
    order : {'C', 'F'}, optional
        Whether to store multi-dimensional data in C (row-major) or
        Fortran (column-major) order in memory.
    chunks : int or tuple of int, optional
        Chunks for Dask. Will be approximated if not set.

    Returns
    -------
    out : dask array
        Array of uninitialized (arbitrary) data with the given
        shape, dtype, and order.
    """
    if chunks is None:
        chunks = est_best_chunk(shape, dtype)

    return da.empty(shape, dtype=dtype, order=order, chunks=chunks)
