import functools
import inspect
import sys
import warnings
from contextlib import contextmanager

import numpy as np

from ._warnings import all_warnings, warn

__all__ = [
    'deprecate_func',
    'get_bound_method_class',
    'all_warnings',
    'safe_as_int',
    'check_shape_equality',
    'check_nD',
    'warn',
    'reshape_nd',
    'identity',
    'slice_at_axis',
    "deprecate_parameter",
    "DEPRECATED",
]


def count_inner_wrappers(func):
    """Count the number of inner wrappers by unpacking ``__wrapped__``.

    If a wrapped function wraps another wrapped function, then we refer to the
    wrapping of the second function as an *inner wrapper*.

    For example, consider this code fragment:

    .. code-block:: python
        @wrap_outer
        @wrap_inner
        def foo():
            pass

    Here ``@wrap_inner`` applies a wrapper to ``foo``, and ``@wrap_outer``
    applies a wrapper to the result.

    Parameters
    ----------
    func : callable
        The callable of which to determine the number of inner wrappers.

    Returns
    -------
    count : int
        The number of times `func` has been wrapped.

    See Also
    --------
    count_global_wrappers
    """
    unwrapped = func
    count = 0
    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__
        count += 1
    return count


def _warning_stacklevel(func):
    """Find stacklevel of `func` relative to its global representation.

    Determine automatically with which stacklevel a warning should be raised.

    Parameters
    ----------
    func : Callable
        Tries to find the global version of `func` and counts the number of
        additional wrappers around `func`.

    Returns
    -------
    stacklevel : int
        The stacklevel. Minimum of 2.
    """
    # Count number of wrappers around `func`
    inner_wrapped_count = count_inner_wrappers(func)
    global_wrapped_count = count_global_wrappers(func)

    stacklevel = global_wrapped_count - inner_wrapped_count + 1
    return max(stacklevel, 2)


def count_global_wrappers(func):
    """Count the total number of times a function as been wrapped globally.

    Similar to :func:`count_inner_wrappers`, this counts the number of times
    `func` has been wrapped. However, this function doesn't start counting
    from `func` but instead tries to access the "global representation" of
    `func`. This means that you could use this function from inside a wrapper
    that was applied first, and still count wrappers that were applied on
    top of it afterwards.

    E.g., `func` might be wrapped by multiple decorators that emit
    warnings. In that case, calling this function in the inner-most decorator
    will still return the total count of wrappers.

    Parameters
    ----------
    func : callable
        The callable of which to determine the number of wrappers. Can be a
        function or method of a class.

    Returns
    -------
    count : int
        The number of times `func` has been wrapped.

    See Also
    --------
    count_inner_wrappers
    """
    if "<locals>" in func.__qualname__:
        msg = (
            "Cannot determine stacklevel of a function defined in another "
            "function's local namespace. Set the stacklevel manually."
        )
        raise ValueError(msg)

    first_name, *other = func.__qualname__.split(".")
    global_func = func.__globals__.get(first_name, func)

    # Account for `func` being a method, in which case it's an attribute of
    # what we got from `func.__globals__`
    for part in other:
        global_func = getattr(global_func, part, global_func)

    count = count_inner_wrappers(global_func)
    assert count >= 0
    return count


class change_default_value:
    """Decorator for changing the default value of an argument.

    Parameters
    ----------
    arg_name : str
        The name of the argument to be updated.
    new_value : any
        The argument new value.
    changed_version : str
        The package version in which the change will be introduced.
    warning_msg : str
        Optional warning message. If None, a generic warning message
        is used.
    stacklevel : {None, int}, optional
        If None, the decorator attempts to detect the appropriate stacklevel for the
        deprecation warning automatically. This can fail, e.g., due to
        decorating a closure, in which case you can set the stacklevel manually
        here. The outermost decorator should have stacklevel 2, the next inner
        one stacklevel 3, etc.
    """

    def __init__(
        self, arg_name, *, new_value, changed_version, warning_msg=None, stacklevel=None
    ):
        self.arg_name = arg_name
        self.new_value = new_value
        self.warning_msg = warning_msg
        self.changed_version = changed_version
        self.stacklevel = stacklevel

    def __call__(self, func):
        parameters = inspect.signature(func).parameters
        arg_idx = list(parameters.keys()).index(self.arg_name)
        old_value = parameters[self.arg_name].default

        if self.warning_msg is None:
            self.warning_msg = (
                f'The new recommended value for {self.arg_name} is '
                f'{self.new_value}. Until version {self.changed_version}, '
                f'the default {self.arg_name} value is {old_value}. '
                f'From version {self.changed_version}, the {self.arg_name} '
                f'default value will be {self.new_value}. To avoid '
                f'this warning, please explicitly set {self.arg_name} value.'
            )

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            if len(args) < arg_idx + 1 and self.arg_name not in kwargs.keys():
                stacklevel = (
                    self.stacklevel
                    if self.stacklevel is not None
                    else _warning_stacklevel(func)
                )
                # warn that arg_name default value changed:
                warnings.warn(self.warning_msg, FutureWarning, stacklevel=stacklevel)
            return func(*args, **kwargs)

        return fixed_func


class PatchClassRepr(type):
    """Control class representations in rendered signatures."""

    def __repr__(cls):
        return f"<{cls.__name__}>"


class DEPRECATED(metaclass=PatchClassRepr):
    """Signal value to help with deprecating parameters that use None.

    This is a proxy object, used to signal that a parameter has not been set.
    This is useful if ``None`` is already used for a different purpose or just
    to highlight a deprecated parameter in the signature.
    """


class deprecate_parameter:
    """Deprecate a parameter of a function.

    Parameters
    ----------
    deprecated_name : str
        The name of the deprecated parameter.
    start_version : str
        The package version in which the warning was introduced.
    stop_version : str
        The package version in which the warning will be replaced by
        an error / the deprecation is completed.
    template : str, optional
        If given, this message template is used instead of the default one.
    new_name : str, optional
        If given, the default message will recommend the new parameter name and an
        error will be raised if the user uses both old and new names for the
        same parameter.
    modify_docstring : bool, optional
        If the wrapped function has a docstring, add the deprecated parameters
        to the "Other Parameters" section.
    stacklevel : {None, int}, optional
        If None, the decorator attempts to detect the appropriate stacklevel for the
        deprecation warning automatically. This can fail, e.g., due to
        decorating a closure, in which case you can set the stacklevel manually
        here. The outermost decorator should have stacklevel 2, the next inner
        one stacklevel 3, etc.

    Notes
    -----
    Assign `DEPRECATED` as the new default value for the deprecated parameter.
    This marks the status of the parameter also in the signature and rendered
    HTML docs.

    This decorator can be stacked to deprecate more than one parameter.

    Examples
    --------
    >>> from skimage._shared.utils import deprecate_parameter, DEPRECATED
    >>> @deprecate_parameter(
    ...     "b", new_name="c", start_version="0.1", stop_version="0.3"
    ... )
    ... def foo(a, b=DEPRECATED, *, c=None):
    ...     return a, c

    Calling ``foo(1, b=2)``  will warn with::

        FutureWarning: Parameter `b` is deprecated since version 0.1 and will
        be removed in 0.3 (or later). To avoid this warning, please use the
        parameter `c` instead. For more details, see the documentation of
        `foo`.
    """

    DEPRECATED = DEPRECATED  # Make signal value accessible for convenience

    remove_parameter_template = (
        "Parameter `{deprecated_name}` is deprecated since version "
        "{deprecated_version} and will be removed in {changed_version} (or "
        "later). To avoid this warning, please do not use the parameter "
        "`{deprecated_name}`. For more details, see the documentation of "
        "`{func_name}`."
    )

    replace_parameter_template = (
        "Parameter `{deprecated_name}` is deprecated since version "
        "{deprecated_version} and will be removed in {changed_version} (or "
        "later). To avoid this warning, please use the parameter `{new_name}` "
        "instead. For more details, see the documentation of `{func_name}`."
    )

    def __init__(
        self,
        deprecated_name,
        *,
        start_version,
        stop_version,
        template=None,
        new_name=None,
        modify_docstring=True,
        stacklevel=None,
    ):
        self.deprecated_name = deprecated_name
        self.new_name = new_name
        self.template = template
        self.start_version = start_version
        self.stop_version = stop_version
        self.modify_docstring = modify_docstring
        self.stacklevel = stacklevel

    def __call__(self, func):
        parameters = inspect.signature(func).parameters
        try:
            deprecated_idx = list(parameters.keys()).index(self.deprecated_name)
        except ValueError as e:
            raise ValueError(f"{self.deprecated_name!r} not in parameters") from e

        new_idx = False
        if self.new_name:
            try:
                new_idx = list(parameters.keys()).index(self.new_name)
            except ValueError as e:
                raise ValueError(f"{self.new_name!r} not in parameters") from e

        if parameters[self.deprecated_name].default is not DEPRECATED:
            raise RuntimeError(
                f"Expected `{self.deprecated_name}` to have the value {DEPRECATED!r} "
                f"to indicate its status in the rendered signature."
            )

        if self.template is not None:
            template = self.template
        elif self.new_name is not None:
            template = self.replace_parameter_template
        else:
            template = self.remove_parameter_template
        warning_message = template.format(
            deprecated_name=self.deprecated_name,
            deprecated_version=self.start_version,
            changed_version=self.stop_version,
            func_name=func.__qualname__,
            new_name=self.new_name,
        )

        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            deprecated_value = DEPRECATED
            new_value = DEPRECATED

            # Extract value of deprecated parameter
            if len(args) > deprecated_idx:
                deprecated_value = args[deprecated_idx]
                # Overwrite old with DEPRECATED if replacement exists
                if self.new_name is not None:
                    args = (
                        args[:deprecated_idx]
                        + (DEPRECATED,)
                        + args[deprecated_idx + 1 :]
                    )
            if self.deprecated_name in kwargs.keys():
                deprecated_value = kwargs[self.deprecated_name]
                # Overwrite old with DEPRECATED if replacement exists
                if self.new_name is not None:
                    kwargs[self.deprecated_name] = DEPRECATED

            # Extract value of new parameter (if present)
            if new_idx is not False and len(args) > new_idx:
                new_value = args[new_idx]
            if self.new_name and self.new_name in kwargs.keys():
                new_value = kwargs[self.new_name]

            if deprecated_value is not DEPRECATED:
                stacklevel = (
                    self.stacklevel
                    if self.stacklevel is not None
                    else _warning_stacklevel(func)
                )
                warnings.warn(
                    warning_message, category=FutureWarning, stacklevel=stacklevel
                )

                if new_value is not DEPRECATED:
                    raise ValueError(
                        f"Both deprecated parameter `{self.deprecated_name}` "
                        f"and new parameter `{self.new_name}` are used. Use "
                        f"only the latter to avoid conflicting values."
                    )
                elif self.new_name is not None:
                    # Assign old value to new one
                    kwargs[self.new_name] = deprecated_value

            return func(*args, **kwargs)

        if self.modify_docstring and func.__doc__ is not None:
            newdoc = _docstring_add_deprecated(
                func, {self.deprecated_name: self.new_name}, self.start_version
            )
            fixed_func.__doc__ = newdoc

        return fixed_func


def _docstring_add_deprecated(func, kwarg_mapping, deprecated_version):
    """Add deprecated kwarg(s) to the "Other Params" section of a docstring.

    Parameters
    ----------
    func : function
        The function whose docstring we wish to update.
    kwarg_mapping : dict
        A dict containing {old_arg: new_arg} key/value pairs, see
        `deprecate_parameter`.
    deprecated_version : str
        A major.minor version string specifying when old_arg was
        deprecated.

    Returns
    -------
    new_doc : str
        The updated docstring. Returns the original docstring if numpydoc is
        not available.
    """
    if func.__doc__ is None:
        return None
    try:
        from numpydoc.docscrape import FunctionDoc, Parameter
    except ImportError:
        # Return an unmodified docstring if numpydoc is not available.
        return func.__doc__

    Doc = FunctionDoc(func)
    for old_arg, new_arg in kwarg_mapping.items():
        desc = []
        if new_arg is None:
            desc.append(f'`{old_arg}` is deprecated.')
        else:
            desc.append(f'Deprecated in favor of `{new_arg}`.')

        desc += ['', f'.. deprecated:: {deprecated_version}']
        Doc['Other Parameters'].append(
            Parameter(name=old_arg, type='DEPRECATED', desc=desc)
        )
    new_docstring = str(Doc)

    # new_docstring will have a header starting with:
    #
    # .. function:: func.__name__
    #
    # and some additional blank lines. We strip these off below.
    split = new_docstring.split('\n')
    no_header = split[1:]
    while not no_header[0].strip():
        no_header.pop(0)

    # Store the initial description before any of the Parameters fields.
    # Usually this is a single line, but the while loop covers any case
    # where it is not.
    descr = no_header.pop(0)
    while no_header[0].strip():
        descr += '\n    ' + no_header.pop(0)
    descr += '\n\n'
    # '\n    ' rather than '\n' here to restore the original indentation.
    final_docstring = descr + '\n    '.join(no_header)
    # strip any extra spaces from ends of lines
    final_docstring = '\n'.join([line.rstrip() for line in final_docstring.split('\n')])
    return final_docstring


class FailedEstimationAccessError(AttributeError):
    """Error from use of failed estimation instance

    This error arises from attempts to use an instance of
    :class:`FailedEstimation`.
    """


class FailedEstimation:
    """Class to indicate a failed transform estimation.

    The ``from_estimate`` class method of each transform type may return an
    instance of this class to indicate some failure in the estimation process.

    Parameters
    ----------
    message : str
        Message indicating reason for failed estimation.

    Attributes
    ----------
    message : str
        Message above.

    Raises
    ------
    FailedEstimationAccessError
        Exception raised for missing attributes or if the instance is used as a
        callable.
    """

    error_cls = FailedEstimationAccessError

    hint = (
        "You can check for a failed estimation by truth testing the returned "
        "object. For failed estimations, `bool(estimation_result)` will be `False`. "
        "E.g.\n\n"
        "    if not estimation_result:\n"
        "        raise RuntimeError(f'Failed estimation: {estimation_result}')"
    )

    def __init__(self, message):
        self.message = message

    def __bool__(self):
        return False

    def __repr__(self):
        return f"{type(self).__name__}({self.message!r})"

    def __str__(self):
        return self.message

    def __call__(self, *args, **kwargs):
        msg = (
            f'{type(self).__name__} is not callable. {self.message}\n\n'
            f'Hint: {self.hint}'
        )
        raise self.error_cls(msg)

    def __getattr__(self, name):
        msg = (
            f'{type(self).__name__} has no attribute {name!r}. {self.message}\n\n'
            f'Hint: {self.hint}'
        )
        raise self.error_cls(msg)


@contextmanager
def _ignore_deprecated_estimate_warning():
    """Filter warnings about the deprecated `estimate` method.

    Use either as decorator or context manager.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            action="ignore",
            category=FutureWarning,
            message="`estimate` is deprecated",
            module="skimage",
        )
        yield


class channel_as_last_axis:
    """Decorator for automatically making channels axis last for all arrays.

    This decorator reorders axes for compatibility with functions that only
    support channels along the last axis. After the function call is complete
    the channels axis is restored back to its original position.

    Parameters
    ----------
    channel_arg_positions : tuple of int, optional
        Positional arguments at the positions specified in this tuple are
        assumed to be multichannel arrays. The default is to assume only the
        first argument to the function is a multichannel array.
    channel_kwarg_names : tuple of str, optional
        A tuple containing the names of any keyword arguments corresponding to
        multichannel arrays.
    multichannel_output : bool, optional
        A boolean that should be True if the output of the function is not a
        multichannel array and False otherwise. This decorator does not
        currently support the general case of functions with multiple outputs
        where some or all are multichannel.

    """

    def __init__(
        self,
        channel_arg_positions=(0,),
        channel_kwarg_names=(),
        multichannel_output=True,
    ):
        self.arg_positions = set(channel_arg_positions)
        self.kwarg_names = set(channel_kwarg_names)
        self.multichannel_output = multichannel_output

    def __call__(self, func):
        @functools.wraps(func)
        def fixed_func(*args, **kwargs):
            channel_axis = kwargs.get('channel_axis', None)

            if channel_axis is None:
                return func(*args, **kwargs)

            # TODO: convert scalars to a tuple in anticipation of eventually
            #       supporting a tuple of channel axes. Right now, only an
            #       integer or a single-element tuple is supported, though.
            if np.isscalar(channel_axis):
                channel_axis = (channel_axis,)
            if len(channel_axis) > 1:
                raise ValueError("only a single channel axis is currently supported")

            if channel_axis == (-1,) or channel_axis == -1:
                return func(*args, **kwargs)

            if self.arg_positions:
                new_args = []
                for pos, arg in enumerate(args):
                    if pos in self.arg_positions:
                        new_args.append(np.moveaxis(arg, channel_axis[0], -1))
                    else:
                        new_args.append(arg)
                new_args = tuple(new_args)
            else:
                new_args = args

            for name in self.kwarg_names:
                kwargs[name] = np.moveaxis(kwargs[name], channel_axis[0], -1)

            # now that we have moved the channels axis to the last position,
            # change the channel_axis argument to -1
            kwargs["channel_axis"] = -1

            # Call the function with the fixed arguments
            out = func(*new_args, **kwargs)
            if self.multichannel_output:
                out = np.moveaxis(out, -1, channel_axis[0])
            return out

        return fixed_func


class deprecate_func:
    """Decorate a deprecated function and warn when it is called.

    Adapted from <http://wiki.python.org/moin/PythonDecoratorLibrary>.

    Parameters
    ----------
    deprecated_version : str
        The package version when the deprecation was introduced.
    removed_version : str
        The package version in which the deprecated function will be removed.
    hint : str, optional
        A hint on how to address this deprecation,
        e.g., "Use `skimage.submodule.alternative_func` instead."
    stacklevel :  {None, int}, optional
        If None, the decorator attempts to detect the appropriate stacklevel for the
        deprecation warning automatically. This can fail, e.g., due to
        decorating a closure, in which case you can set the stacklevel manually
        here. The outermost decorator should have stacklevel 2, the next inner
        one stacklevel 3, etc.

    Examples
    --------
    >>> @deprecate_func(
    ...     deprecated_version="1.0.0",
    ...     removed_version="1.2.0",
    ...     hint="Use `bar` instead."
    ... )
    ... def foo():
    ...     pass

    Calling ``foo`` will warn with::

        FutureWarning: `foo` is deprecated since version 1.0.0
        and will be removed in version 1.2.0. Use `bar` instead.
    """

    def __init__(
        self, *, deprecated_version, removed_version=None, hint=None, stacklevel=None
    ):
        self.deprecated_version = deprecated_version
        self.removed_version = removed_version
        self.hint = hint
        self.stacklevel = stacklevel

    def __call__(self, func):
        message = (
            f"`{func.__name__}` is deprecated since version {self.deprecated_version}"
        )
        if self.removed_version:
            message += f" and will be removed in version {self.removed_version}."
        if self.hint:
            # Prepend space and make sure it closes with "."
            message += f" {self.hint.rstrip('.')}."

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            stacklevel = (
                self.stacklevel
                if self.stacklevel is not None
                else _warning_stacklevel(func)
            )
            warnings.warn(message, category=FutureWarning, stacklevel=stacklevel)
            return func(*args, **kwargs)

        # modify docstring to display deprecation warning
        doc = f'**Deprecated:** {message}'
        if wrapped.__doc__ is None:
            wrapped.__doc__ = doc
        else:
            wrapped.__doc__ = doc + '\n\n    ' + wrapped.__doc__

        return wrapped


def _deprecate_estimate(func, class_name=None):
    """Deprecate ``estimate`` method."""
    class_name = func.__qualname__.split('.')[0] if class_name is None else class_name
    return deprecate_func(
        deprecated_version="0.26",
        removed_version="2.2",
        hint=f"Please use `{class_name}.from_estimate` class constructor instead.",
        stacklevel=2,
    )(func)


def _deprecate_inherited_estimate(cls):
    """Deprecate inherited ``estimate`` instance method.

    This needs a class decorator so we can correctly specify the class of the
    `from_estimate` class method in the deprecation message.
    """

    def estimate(self, *args, **kwargs):
        return self._estimate(*args, **kwargs) is None

    # The inherited method will always be wrapped by deprecator.
    inherited_meth = getattr(cls, 'estimate').__wrapped__
    estimate.__doc__ = inherited_meth.__doc__
    estimate.__signature__ = inspect.signature(inherited_meth)

    cls.estimate = _deprecate_estimate(estimate, cls.__name__)
    return cls


def _update_from_estimate_docstring(cls):
    """Fix docstring for inherited ``from_estimate`` class method.

    Even for classes that inherit the `from_estimate` method, and do not
    override it, we nevertheless need to change the *docstring* of the
    `from_estimate` method to point the user to the current (inheriting) class,
    rather than the class in which the method is defined (the inherited class).

    This needs a class decorator so we can modify the docstring of the new
    class method.  CPython currently does not allow us to modify class method
    docstrings by updating ``__doc__``.
    """

    inherited_cmeth = getattr(cls, 'from_estimate')

    def from_estimate(cls, *args, **kwargs):
        return inherited_cmeth(*args, **kwargs)

    inherited_class_name = inherited_cmeth.__qualname__.split('.')[-2]

    from_estimate.__doc__ = inherited_cmeth.__doc__.replace(
        inherited_class_name, cls.__name__
    )
    from_estimate.__signature__ = inspect.signature(inherited_cmeth)

    cls.from_estimate = classmethod(from_estimate)
    return cls


def get_bound_method_class(m):
    """Return the class for a bound method."""
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
    mod = np.asarray(val) % 1  # Extract mantissa

    # Check for and subtract any mod values > 0.5 from 1
    if mod.ndim == 0:  # Scalar input, cannot be indexed
        if mod > 0.5:
            mod = 1 - mod
    else:  # Iterable input, now ndarray
        mod[mod > 0.5] = 1 - mod[mod > 0.5]  # Test on each side of nearest int

    if not np.allclose(mod, 0, atol=atol):
        raise ValueError(f'Integer argument required but received {val}, check inputs.')

    return np.round(val).astype(np.int64)


def check_shape_equality(*images):
    """Check that all images have the same shape"""
    image0 = images[0]
    if not all(image0.shape == image.shape for image in images[1:]):
        raise ValueError('Input images must have the same dimensions.')
    return


def slice_at_axis(sl, axis):
    """
    Construct tuple of slices to slice an array in the given dimension.

    Parameters
    ----------
    sl : slice
        The slice for the given dimension.
    axis : int
        The axis to which `sl` is applied. All other dimensions are left
        "unsliced".

    Returns
    -------
    sl : tuple of slices
        A tuple with slices matching `shape` in length.

    Examples
    --------
    >>> slice_at_axis(slice(None, 3, -1), 1)
    (slice(None, None, None), slice(None, 3, -1), Ellipsis)
    """
    return (slice(None),) * axis + (sl,) + (...,)


def reshape_nd(arr, ndim, dim):
    """Reshape a 1D array to have n dimensions, all singletons but one.

    Parameters
    ----------
    arr : array, shape (N,)
        Input array
    ndim : int
        Number of desired dimensions of reshaped array.
    dim : int
        Which dimension/axis will not be singleton-sized.

    Returns
    -------
    arr_reshaped : array, shape ([1, ...], N, [1,...])
        View of `arr` reshaped to the desired shape.

    Examples
    --------
    >>> rng = np.random.default_rng()
    >>> arr = rng.random(7)
    >>> reshape_nd(arr, 2, 0).shape
    (7, 1)
    >>> reshape_nd(arr, 3, 1).shape
    (1, 7, 1)
    >>> reshape_nd(arr, 4, -1).shape
    (1, 1, 1, 7)
    """
    if arr.ndim != 1:
        raise ValueError("arr must be a 1D array")
    new_shape = [1] * ndim
    new_shape[dim] = -1
    return np.reshape(arr, new_shape)


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
    if array.ndim not in ndim:
        raise ValueError(
            msg_incorrect_dim % (arg_name, '-or-'.join([str(n) for n in ndim]))
        )


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

    Notes
    -----
    * Input images with `float32` data type are not upcast.

    Returns
    -------
    image : ndarray
        Transformed version of the input.

    """
    if image.dtype == np.float16:
        return image.astype(np.float32)
    if preserve_range:
        # Convert image to double only if it is not single or double
        # precision float
        if image.dtype.char not in 'df':
            image = image.astype(float)
    else:
        from ..util.dtype import img_as_float

        image = img_as_float(image)
    return image


def _validate_interpolation_order(image_dtype, order):
    """Validate and return spline interpolation's order.

    Parameters
    ----------
    image_dtype : dtype
        Image dtype.
    order : {None, int}, optional
        The order of the spline interpolation. The order has to be in the range
        0-5. If ``None`` assume order 0 for Boolean images, otherwise 1. See
        `skimage.transform.warp` for detail.

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
        raise ValueError("Spline interpolation order has to be in the range 0-5.")

    if image_dtype == bool and order != 0:
        raise ValueError(
            "Input image dtype is bool. Interpolation is not defined "
            "with bool data type. Please set order to 0 or explicitly "
            "cast input image to another data type."
        )

    return order


def _to_np_mode(mode):
    """Convert padding modes from `ndi.correlate` to `np.pad`."""
    mode_translation_dict = dict(nearest='edge', reflect='symmetric', mirror='reflect')
    if mode in mode_translation_dict:
        mode = mode_translation_dict[mode]
    return mode


def _to_ndimage_mode(mode):
    """Convert from `numpy.pad` mode name to the corresponding ndimage mode."""
    mode_translation_dict = dict(
        constant='constant',
        edge='nearest',
        symmetric='reflect',
        reflect='mirror',
        wrap='wrap',
    )
    if mode not in mode_translation_dict:
        raise ValueError(
            f"Unknown mode: '{mode}', or cannot translate mode. The "
            f"mode should be one of 'constant', 'edge', 'symmetric', "
            f"'reflect', or 'wrap'. See the documentation of numpy.pad for "
            f"more info."
        )
    return _fix_ndimage_mode(mode_translation_dict[mode])


def _fix_ndimage_mode(mode):
    # SciPy 1.6.0 introduced grid variants of constant and wrap which
    # have less surprising behavior for images. Use these when available
    grid_modes = {'constant': 'grid-constant', 'wrap': 'grid-wrap'}
    return grid_modes.get(mode, mode)


new_float_type = {
    # preserved types
    np.float32().dtype.char: np.float32,
    np.float64().dtype.char: np.float64,
    np.complex64().dtype.char: np.complex64,
    np.complex128().dtype.char: np.complex128,
    # altered types
    np.float16().dtype.char: np.float32,
    'g': np.float64,  # np.float128 ; doesn't exist on windows
    'G': np.complex128,  # np.complex256 ; doesn't exist on windows
}


def _supported_float_type(input_dtype, allow_complex=False):
    """Return an appropriate floating-point dtype for a given dtype.

    float32, float64, complex64, complex128 are preserved.
    float16 is promoted to float32.
    complex256 is demoted to complex128.
    Other types are cast to float64.

    Parameters
    ----------
    input_dtype : np.dtype or tuple of np.dtype
        The input dtype. If a tuple of multiple dtypes is provided, each
        dtype is first converted to a supported floating point type and the
        final dtype is then determined by applying `np.result_type` on the
        sequence of supported floating point types.
    allow_complex : bool, optional
        If False, raise a ValueError on complex-valued inputs.

    Returns
    -------
    float_type : dtype
        Floating-point dtype for the image.
    """
    if isinstance(input_dtype, tuple):
        return np.result_type(*(_supported_float_type(d) for d in input_dtype))
    input_dtype = np.dtype(input_dtype)
    if not allow_complex and input_dtype.kind == 'c':
        raise ValueError("complex valued input is not supported")
    return new_float_type.get(input_dtype.char, np.float64)


def identity(image, *args, **kwargs):
    """Returns the first argument unmodified."""
    return image


def as_binary_ndarray(array, *, variable_name):
    """Return `array` as a numpy.ndarray of dtype bool.

    Raises
    ------
    ValueError:
        An error including the given `variable_name` if `array` can not be
        safely cast to a boolean array.
    """
    array = np.asarray(array)
    if array.dtype != bool:
        if np.any((array != 1) & (array != 0)):
            raise ValueError(
                f"{variable_name} array is not of dtype boolean or "
                f"contains values other than 0 and 1 so cannot be "
                f"safely cast to boolean array."
            )
    return np.asarray(array, dtype=bool)
