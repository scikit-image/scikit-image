import numpy as np


def numeric_dtype_min_max(dtype):
    """Return minimum and maximum representable value for a given dtype.

    A convenient wrapper around `numpy.finfo` and `numpy.iinfo` that
    additionally supports numpy.bool as well.

    Parameters
    ----------
    dtype : numpy.dtype
        The dtype. Tries to convert Python "types" such as int or float, to
        the corresponding NumPy dtype.

    Returns
    -------
    min, max : number
        Minimum and maximum of the given `dtype`. These scalars are themselves
        of the given `dtype`.

    Examples
    --------
    >>> import numpy as np
    >>> numeric_dtype_min_max(np.uint8)
    (0, 255)
    >>> numeric_dtype_min_max(bool)
    (False, True)
    >>> numeric_dtype_min_max(np.float64)
    (-1.7976931348623157e+308, 1.7976931348623157e+308)
    >>> numeric_dtype_min_max(int)
    (-9223372036854775808, 9223372036854775807)
    """
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        min_ = dtype.type(info.min)
        max_ = dtype.type(info.max)
    elif np.issubdtype(dtype, np.inexact):
        info = np.finfo(dtype)
        min_ = info.min
        max_ = info.max
    elif np.issubdtype(dtype, np.dtype(bool)):
        min_ = dtype.type(False)
        max_ = dtype.type(True)
    else:
        raise ValueError(f"unsupported dtype {dtype!r}")
    return min_, max_
