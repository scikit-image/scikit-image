import inspect
import warnings
import functools
import sys
import numpy as np
import numbers

from ..util import img_as_float
from ._warnings import all_warnings, warn

__all__ = ['deprecated', 'get_bound_method_class', 'all_warnings',
           'safe_as_int', 'check_nD', 'check_shape_equality', 'warn']


class skimage_deprecation(Warning):
    """Create our own deprecation class, since Python >= 2.7
    silences deprecations by default.

    """
    pass


class change_default_value:
    """Decorator for changing the default value of an argument.

    Parameters
    ----------
    arg_name: str
        The name of the argument to be updated.
    new_value: any
        The argument new value.
    changed_version : str
        The package version in which the change will be introduced.
    warning_msg: str
        Optional warning message. If None, a generic warning message
        is used.

    """

    def __init__(self, arg_name, *, new_value, changed_version,
                 warning_msg=None):
        self.arg_name = arg_name
        self.new_value = new_value
        self.warning_msg = warning_msg
        self.changed_version = changed_version

    def __call__(self, func):
        parameters = inspect.signature(func).parameters
        arg_idx = list(parameters.keys()).index(self.arg_name)
        old_value = parameters[self.arg_name].default

        if self.warning_msg is None:
            self.warning_msg = (
                f"The new recommended value for {self.arg_name} is "
                f"{self.new_value}. Until version {self.changed_version}, "
                f"the default {self.arg_name} value is {old_value}. "
                f"From version {self.changed_version}, the {self.arg_name} "
                f"default value will be {self.new_value}. To avoid "
                f"this warning, please explicitly set {self.arg_name} value.")

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            if len(args) < arg_idx + 1 and self.arg_name not in kwargs.keys():
                # warn that arg_name default value changed:
                warnings.warn(self.warning_msg, FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        return fixed_func


class remove_arg:
    """Decorator to remove an argument from function's signature.

    Parameters
    ----------
    arg_name: str
        The name of the argument to be removed.
    changed_version : str
        The package version in which the warning will be replaced by
        an error.
    help_msg: str
        Optional message appended to the generic warning message.

    """

    def __init__(self, arg_name, *, changed_version, help_msg=None):
        self.arg_name = arg_name
        self.help_msg = help_msg
        self.changed_version = changed_version

    def __call__(self, func):
        parameters = inspect.signature(func).parameters
        arg_idx = list(parameters.keys()).index(self.arg_name)
        warning_msg = (
            f"{self.arg_name} argument is deprecated and will be removed "
            f"in version {self.changed_version}. To avoid this warning, "
            f"please do not use the {self.arg_name} argument. Please "
            f"see {func.__name__} documentation for more details.")

        if self.help_msg is not None:
            warning_msg += f" {self.help_msg}"

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            if len(args) > arg_idx or self.arg_name in kwargs.keys():
                # warn that arg_name is deprecated
                warnings.warn(warning_msg, FutureWarning, stacklevel=2)
            return func(*args, **kwargs)

        return fixed_func


class deprecate_kwarg:
    """Decorator ensuring backward compatibility when argument names are
    modified in a function definition.

    Parameters
    ----------
    arg_mapping: dict
        Mapping between the function's old argument names and the new
        ones.
    warning_msg: str
        Optional warning message. If None, a generic warning message
        is used.
    removed_version : str
        The package version in which the deprecated argument will be
        removed.

    """

    def __init__(self, kwarg_mapping, warning_msg=None, removed_version=None):
        self.kwarg_mapping = kwarg_mapping
        if warning_msg is None:
            self.warning_msg = ("'{old_arg}' is a deprecated argument name "
                                "for `{func_name}`. ")
            if removed_version is not None:
                self.warning_msg += ("It will be removed in version {}. "
                                     .format(removed_version))
            self.warning_msg += "Please use '{new_arg}' instead."
        else:
            self.warning_msg = warning_msg

    def __call__(self, func):
        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            for old_arg, new_arg in self.kwarg_mapping.items():
                if old_arg in kwargs:
                    #  warn that the function interface has changed:
                    warnings.warn(self.warning_msg.format(
                        old_arg=old_arg, func_name=func.__name__,
                        new_arg=new_arg), FutureWarning, stacklevel=2)
                    # Substitute new_arg to old_arg
                    kwargs[new_arg] = kwargs.pop(old_arg)

            # Call the function with the fixed arguments
            return func(*args, **kwargs)
        return fixed_func


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
    removed_version : str
        The package version in which the deprecated function will be removed.
    """

    def __init__(self, alt_func=None, behavior='warn', removed_version=None):
        self.alt_func = alt_func
        self.behavior = behavior
        self.removed_version = removed_version

    def __call__(self, func):

        alt_msg = ''
        if self.alt_func is not None:
            alt_msg = ' Use ``%s`` instead.' % self.alt_func
        rmv_msg = ''
        if self.removed_version is not None:
            rmv_msg = (' and will be removed in version %s' %
                       self.removed_version)

        msg = ('Function ``%s`` is deprecated' % func.__name__ +
               rmv_msg + '.' + alt_msg)

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if self.behavior == 'warn':
                func_code = func.__code__
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
    >>> safe_as_int(7.0)
    7

    >>> safe_as_int([9, 4, 2.9999999999])
    array([9, 4, 3])

    >>> safe_as_int(53.1)
    Traceback (most recent call last):
        ...
    ValueError: Integer argument required but received 53.1, check inputs.

    >>> safe_as_int(53.01, atol=0.01)
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


def check_shape_equality(im1, im2):
    """Raise an error if the shape do not match."""
    if not im1.shape == im2.shape:
        raise ValueError('Input images must have the same dimensions.')
    return


def check_nD(array, ndim, arg_name='image'):
    """
    Verify an array meets the desired ndims and array isn't empty.

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
    msg_incorrect_dim = "The parameter `%s` must be a %s-dimensional array"
    msg_empty_array = "The parameter `%s` cannot be an empty array"
    if isinstance(ndim, int):
        ndim = [ndim]
    if array.size == 0:
        raise ValueError(msg_empty_array % (arg_name))
    if not array.ndim in ndim:
        raise ValueError(msg_incorrect_dim % (arg_name, '-or-'.join([str(n) for n in ndim])))


def check_random_state(seed):
    """Turn seed into a `np.random.RandomState` instance.

    Parameters
    ----------
    seed : None, int or np.random.RandomState
           If `seed` is None, return the RandomState singleton used by `np.random`.
           If `seed` is an int, return a new RandomState instance seeded with `seed`.
           If `seed` is already a RandomState instance, return it.

    Raises
    ------
    ValueError
        If `seed` is of the wrong type.

    """
    # Function originally from scikit-learn's module sklearn.utils.validation
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def convert_to_float(image, preserve_range):
    """Convert input image to float image with the appropriate range.

    Parameters
    ----------
    image : ndarray
        Input image.
    preserve_range : bool
        Determines if the range of the image should be kept or transformed
        using img_as_float. Also see
        https://scikit-image.org/docs/dev/user_guide/data_types.html

    Notes:
    ------
    * Input images with `float32` data type are not upcast.

    Returns
    -------
    image : ndarray
        Transformed version of the input.

    """
    if preserve_range:
        # Convert image to double only if it is not single or double
        # precision float
        if image.dtype.char not in 'df':
            image = image.astype(float)
    else:
        image = img_as_float(image)
    return image


def _validate_interpolation_order(image_dtype, order):
    """Validate and return spline interpolation's order.

    Parameters
    ----------
    image_dtype : dtype
        Image dtype.
    order : int, optional
        The order of the spline interpolation. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.

    Returns
    -------
    order : int
        if input order is None, returns 0 if image_dtype is bool and 1
        otherwise. Otherwise, image_dtype is checked and input order
        is validated accordingly (order > 0 is not supported for bool
        image dtype)

    """

    if order is None:
        return 0 if image_dtype == bool else 1

    if order < 0 or order > 5:
        raise ValueError("Spline interpolation order has to be in the "
                         "range 0-5.")

    if image_dtype == bool and order != 0:
        warn("Input image dtype is bool. Interpolation is not defined "
             "with bool data type. Please set order to 0 or explicitely "
             "cast input image to another data type. Starting from version "
             "0.19 a ValueError will be raised instead of this warning.",
             FutureWarning, stacklevel=2)

    return order
