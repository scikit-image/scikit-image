import os

import numpy as np

import skimage


def _channel_kwarg(is_multichannel=False):
    if np.lib.NumpyVersion(skimage.__version__) < '0.19.0':
        return dict(multichannel=is_multichannel)
    else:
        return dict(channel_axis=-1 if is_multichannel else None)


def _skip_slow():
    """
    Use this function to skip slow or highly demanding tests.

    Use it as a `Class.setup` method or a `function.setup` attribute.

    For example:

    >>> from . import _skip_slow
    >>> def time_something_slow():
    ...     pass
    >>> time_something.setup = _skip_slow
    """
    if os.environ.get("ASV_SKIP_SLOW", "0") == "1":
        raise NotImplementedError("Skipping this test...")


def _resolve_params(reduced, full):
    """
    Pick between a reduced (fast CI) and full (nightly) parameter list.

    Controlled by the ASV_FULL_PARAMS environment variable. Use this to
    wrap one axis of a benchmark class's ``params`` where the reduced
    set trims a heavily-parametrized suite for fast CI runs, while
    nightly runs (ASV_FULL_PARAMS=1) exercise the full original
    parameter space.

    For example:

    >>> from . import _resolve_params
    >>> class Suite:
    ...     params = [_resolve_params(reduced=[1, 2], full=[1, 2, 3, 4, 5])]
    """
    if os.environ.get("ASV_FULL_PARAMS", "0") == "1":
        return full
    return reduced
