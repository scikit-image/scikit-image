import numpy as np


def check_array(arg_pos, kwarg_name, channels=None, dtype=None):
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
                if (
                    (channels > 1 and array.ndim == 2)
                    or (array.ndim == 3 and array.shape[2] != channels)
                ):
                    error_msg = 'invalid number of channels'

            if dtype is not None:
                if np.dtype(dtype) != array.dtype:
                    error_msg = 'invalid dtype'

            if error_msg is not None:
                raise ValueError(error_msg + ' for %s' % image_desc)

            return func(*args, **kwargs)
        return inner
    return wrapper
