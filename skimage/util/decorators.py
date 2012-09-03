import numpy as np
from skimage.util.dtype import convert


def check_array(arg_pos, kwarg_name, channels=None, dtype=None, convert=False):
    """Decorator to check input parameters of a function.

    If the input matches the specified conditions the decorated function is
    called, otherwise a ``ValueError`` is raised.

    Parameters
    ----------
    arg_pos : int
        Position of argument.
    kwarg_name : str
        Name of argument.
    channels : int
        Check array for number of channels. Default is None.
    dtype : str or ``numpy.dtype``
        Check array for specific dtype. Default is None.
    convert : bool
        Automatically convert image to specified `dtype`. Default is False.

    """

    def wrapper(func):
        def inner(*args, **kwargs):

            if kwarg_name in kwargs:
                array = kwargs[kwarg_name]
            else:
                array = args[arg_pos]

            image_desc = 'parameter `%s`' % kwarg_name

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
                        array = convert(array, dtype)
                    else:
                        error_msg = 'invalid dtype'

            if error_msg is not None:
                raise ValueError(error_msg + ' for %s' % image_desc)

            # pass updated array
            if kwarg_name in kwargs:
                kwargs[kwarg_name] = array
            else:
                args = list(args)
                args[arg_pos] = array

            return func(*args, **kwargs)

        return inner
    return wrapper
