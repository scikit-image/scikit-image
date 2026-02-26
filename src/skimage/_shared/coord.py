from textwrap import dedent

import numpy as np

from ..util import PendingSkimage2Change
from .._shared._warnings import warn_external

import skimage2 as ski2


def ensure_spacing(
    coords,
    spacing=1,
    p_norm=np.inf,
    min_split_size=50,
    max_out=None,
    *,
    max_split_size=2000,
):
    """Returns a subset of coord where a minimum spacing is guaranteed.

    Parameters
    ----------
    coords : array_like
        The coordinates of the considered points.
    spacing : float
        the maximum allowed spacing between the points.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.
    min_split_size : int
        Minimum split size used to process ``coords`` by batch to save
        memory. If None, the memory saving strategy is not applied.
    max_out : int
        If not None, only the first ``max_out`` candidates are returned.
    max_split_size : int
        Maximum split size used to process ``coords`` by batch to save
        memory. This number was decided by profiling with a large number
        of points. Too small a number results in too much looping in
        Python instead of C, slowing down the process, while too large
        a number results in large memory allocations, slowdowns, and,
        potentially, in the process being killed -- see gh-6010. See
        benchmark results `here
        <https://github.com/scikit-image/scikit-image/pull/6035#discussion_r751518691>`_.

    Returns
    -------
    output : array_like
        A subset of coord where a minimum spacing is guaranteed.

    """
    warn_external(
        dedent("""\
        `skimage.feature.ensure_spacing` is deprecated in favor of
        `skimage2.feature.ensure_spacing` with new behavior:

        * ...

        To keep the old behavior when switching to `skimage2`, update your call
        according to the following cases:

        * ...

        Other keyword parameters can be left unchanged.
        """),
        category=PendingSkimage2Change,
    )
    output = ski2.feature.ensure_spacing(
        coords=coords,
        spacing=spacing,
        p_norm=p_norm,
        min_split_size=min_split_size,
        max_out=max_out,
        max_split_size=max_split_size,
    )
    return output
