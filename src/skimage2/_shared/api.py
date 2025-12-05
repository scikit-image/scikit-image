"""API manipulation and deprecation helpers."""

from functools import update_wrapper


def copy_interface(src_func, /):
    """Decorator factory to copy docstring and signature between functions.

    Parameters
    ----------
    src_func : Callable


    Returns
    -------
    decorator : Callable
        A custom decorator.


    Examples
    --------
    >>> def _foo_implementation(a, *, b=3):
    ...     '''Foobulate `a` with `b`.'''
    ...     return a + b

    >>> @copy_interface(_foo_implementation)
    ... def foo(*args, **kwargs):
    ...     return _foo_implementation(*args, **kwargs)

    >>> help(foo)
    Help on function foo in module skimage2._shared.api:
    <BLANKLINE>
    foo(a, *, b=3)
        Foobulate `a` with `b`.
    <BLANKLINE>
    """

    def decorator(dst_func):
        update_wrapper(dst_func, src_func, assigned=("__doc__",))
        return dst_func

    return decorator
