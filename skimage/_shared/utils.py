import sys
import warnings
import functools

import numpy as np
from numpy import math as npmath


__all__ = ['deprecated', 'numexpr_eval_fallback']


try:
    import numexpr
except ImportError:
    numexpr = None


def numexpr_eval_fallback(ex, local_dict=None, global_dict=None,
                          out=None, order='K', casting='safe', fallback=False):
    """Call ``numexpr.evaluate`` if existing, otherwise fallback to pure NumPy.

    Parameters
    ----------
    See ``numexpr.evaluate`` for parameter description.

    Note, that the ``casting`` parameter is ignored if ``numexpr`` is not
    available.

    """
    call_frame = sys._getframe(1)
    if local_dict is None:
        local_dict = call_frame.f_locals
    if global_dict is None:
        if numexpr is None or fallback:
            global_dict = npmath.__dict__
        else:
            global_dict = dict()
        global_dict.update(call_frame.f_globals)

    if numexpr is None or fallback:
        print global_dict['sqrt']
        out_temp = eval(ex, global_dict, local_dict)
        out_temp = np.array(out_temp, copy=False, order=order)
        if out is None:
            out = out_temp
        else:
            out[:] = out_temp
    else:
        out = numexpr.evaluate(ex, local_dict=local_dict,
                               global_dict=global_dict, out=out, order=order)

    return out


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
            alt_msg = ' Use `%s` instead.' % self.alt_func

        msg = 'Call to deprecated function `%s`.' % func.__name__
        msg += alt_msg

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if self.behavior == 'warn':
                warnings.warn_explicit(msg,
                    category=DeprecationWarning,
                    filename=func.func_code.co_filename,
                    lineno=func.func_code.co_firstlineno + 1)
            elif self.behavior == 'raise':
                raise DeprecationWarning(msg)
            return func(*args, **kwargs)

        # modify doc string to display deprecation warning
        doc = '**Deprecated function**.' + alt_msg
        if wrapped.__doc__ is None:
            wrapped.__doc__ = doc
        else:
            wrapped.__doc__ = doc + '\n\n' + wrapped.__doc__

        return wrapped
