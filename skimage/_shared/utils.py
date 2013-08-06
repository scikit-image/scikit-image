import warnings
import functools
import sys

from . import six


__all__ = ['deprecated', 'cached_property', 'get_bound_method_class']


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
                warnings.warn_explicit(msg,
                    category=DeprecationWarning,
                    filename=func_code.co_filename,
                    lineno=func_code.co_firstlineno + 1)
            elif self.behavior == 'raise':
                raise DeprecationWarning(msg)
            return func(*args, **kwargs)

        # modify doc string to display deprecation warning
        doc = '**Deprecated function**.' + alt_msg
        if wrapped.__doc__ is None:
            wrapped.__doc__ = doc
        else:
            wrapped.__doc__ = doc + '\n\n    ' + wrapped.__doc__

        return wrapped


class cached_property(object):
    """Decorator to use a function as a cached property.

    The function is only called the first time and each successive call returns
    the cached result of the first call.

        class Foo(object):

            @cached_property
            def foo(self):
                return "Cached"

    Adapted from <http://wiki.python.org/moin/PythonDecoratorLibrary>.

    """

    def __init__(self, func, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, _missing)
        if value is _missing:
            value = self.func(obj)
            obj.__dict__[self.__name__] = value
        return value


def get_bound_method_class(m):
    """Return the class for a bound method.

    """
    return m.im_class if sys.version < '3' else m.__self__.__class__
