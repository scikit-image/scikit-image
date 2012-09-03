import numpy as np
from skimage.util.dtype import convert as convert_func


def check_array(name=None, channels=None, dtype=None, convert=True):
    """Decorator to check input parameters of a function.

    If the input matches the specified conditions the decorated function is
    called, otherwise a ``ValueError`` is raised.

    Parameters
    ----------
    name : str
        Name of argument. Default behaviour is to use the first argument (None).
    channels : int
        Check array for number of channels. Default is None.
    dtype : str or ``numpy.dtype``
        Check array for specific dtype. Default is None.
    convert : bool
        Automatically convert image to specified `dtype`. Default is True.

    """

    def wrapper(func):
        def inner(*args, **kwargs):

            if arg_name in kwargs:
                array = kwargs[arg_name]
            elif len(args) > arg_pos:
                array = args[arg_pos]
            else:
                array = None

            if array is None:
                return func(*args, **kwargs)

            image_desc = 'parameter `%s`' % arg_name

            error_msg = None

            if channels is not None:
                if array.ndim > 2:
                    array_channels = array.shape[2]
                else:
                    array_channels = 1
                if array_channels != channels or array.ndim not in (2, 3):
                    error_msg = 'invalid number of channels'

            if dtype is not None:
                if np.dtype(dtype) != array.dtype:
                    if convert:
                        array = convert_func(array, dtype)
                    else:
                        error_msg = 'invalid dtype'

            if error_msg is not None:
                raise ValueError(error_msg + ' for %s' % image_desc)

            # pass updated array
            if arg_name in kwargs:
                kwargs[arg_name] = array
            else:
                args = list(args)
                args[arg_pos] = array

            return func(*args, **kwargs)

        decorated_func = func
        if hasattr(decorated_func, '__decorated_func'):
            decorated_func = decorated_func.__decorated_func

        if name is None:
            arg_name = decorated_func.func_code.co_varnames[0]
        else:
            arg_name = name

        arg_pos = decorated_func.func_code.co_varnames.index(arg_name)

        # copy function signature
        inner.__name__ = func.__name__
        inner.__doc__ = func.__doc__
        inner.__decorated_func = decorated_func

        return inner
    return wrapper
