import warnings
import functools
import sys
import numpy as np

import six

from ._warnings import all_warnings

__all__ = ['deprecated', 'get_bound_method_class', 'all_warnings',
           'safe_as_int', 'assert_nD']


class skimage_deprecation(Warning):
    """Create our own deprecation class, since Python >= 2.7
    silences deprecations by default.

    """
    pass


class deprecated(object):
    """Decorator to mark deprecated functions with warning.

    Adapted from <http://wiki.python.org/moin/PythonDecoratorLibrary>.

    Parameters
    ----------
    alt_func : str
        If given, tell user what function to use instead.
    behavior : {'warn', 'raise'}
        Behavior during call to deprecated function: 'warn' = warn user that
        function is deprecated; 'raise' = raise error.
    """

    def __init__(self, alt_func=None, behavior='warn'):
        self.alt_func = alt_func
        self.behavior = behavior

    def __call__(self, func):

        alt_msg = ''
        if self.alt_func is not None:
            alt_msg = ' Use ``%s`` instead.' % self.alt_func

        msg = 'Call to deprecated function ``%s``.' % func.__name__
        msg += alt_msg

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if self.behavior == 'warn':
                func_code = six.get_function_code(func)
                warnings.simplefilter('always', skimage_deprecation)
                warnings.warn_explicit(msg,
                                       category=skimage_deprecation,
                                       filename=func_code.co_filename,
                                       lineno=func_code.co_firstlineno + 1)
            elif self.behavior == 'raise':
                raise skimage_deprecation(msg)
            return func(*args, **kwargs)

        # modify doc string to display deprecation warning
        doc = '**Deprecated function**.' + alt_msg
        if wrapped.__doc__ is None:
            wrapped.__doc__ = doc
        else:
            wrapped.__doc__ = doc + '\n\n    ' + wrapped.__doc__

        return wrapped


def get_bound_method_class(m):
    """Return the class for a bound method.

    """
    return m.im_class if sys.version < '3' else m.__self__.__class__


def safe_as_int(val, atol=1e-3):
    """
    Attempt to safely cast values to integer format.

    Parameters
    ----------
    val : scalar or iterable of scalars
        Number or container of numbers which are intended to be interpreted as
        integers, e.g., for indexing purposes, but which may not carry integer
        type.
    atol : float
        Absolute tolerance away from nearest integer to consider values in
        ``val`` functionally integers.

    Returns
    -------
    val_int : NumPy scalar or ndarray of dtype `np.int64`
        Returns the input value(s) coerced to dtype `np.int64` assuming all
        were within ``atol`` of the nearest integer.

    Notes
    -----
    This operation calculates ``val`` modulo 1, which returns the mantissa of
    all values. Then all mantissas greater than 0.5 are subtracted from one.
    Finally, the absolute tolerance from zero is calculated. If it is less
    than ``atol`` for all value(s) in ``val``, they are rounded and returned
    in an integer array. Or, if ``val`` was a scalar, a NumPy scalar type is
    returned.

    If any value(s) are outside the specified tolerance, an informative error
    is raised.

    Examples
    --------
    >>> _safe_as_int(7.0)
    7

    >>> _safe_as_int([9, 4, 2.9999999999])
    array([9, 4, 3], dtype=int32)

    >>> _safe_as_int(53.01)
    Traceback (most recent call last):
        ...
    ValueError: Integer argument required but received 53.1, check inputs.

    >>> _safe_as_int(53.01, atol=0.01)
    53

    """
    mod = np.asarray(val) % 1                # Extract mantissa

    # Check for and subtract any mod values > 0.5 from 1
    if mod.ndim == 0:                        # Scalar input, cannot be indexed
        if mod > 0.5:
            mod = 1 - mod
    else:                                    # Iterable input, now ndarray
        mod[mod > 0.5] = 1 - mod[mod > 0.5]  # Test on each side of nearest int

    try:
        np.testing.assert_allclose(mod, 0, atol=atol)
    except AssertionError:
        raise ValueError("Integer argument required but received "
                         "{0}, check inputs.".format(val))

    return np.round(val).astype(np.int64)


def assert_nD(array, ndim, arg_name='image'):
    """
    Verify an array meets the desired ndims.

    Parameters
    ----------
    array : array-like
        Input array to be validated
    ndim : int or iterable of ints
        Allowable ndim or ndims for the array.
    arg_name : str, optional
        The name of the array in the original function.

    """
    array = np.asanyarray(array)
    msg = "The parameter `%s` must be a %s-dimensional array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if not array.ndim in ndim:
        raise ValueError(msg % (arg_name, '-or-'.join([str(n) for n in ndim])))
