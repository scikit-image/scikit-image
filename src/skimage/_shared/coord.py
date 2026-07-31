import numpy as np

from _skimage2.feature._peaks import _ensure_spacing


def ensure_spacing(
    coords,
    spacing=1,
    p_norm=np.inf,
    min_split_size=50,
    max_out=None,
    *,
    max_split_size=2000,
):
    """Return a subset of coord where a minimum spacing is guaranteed.

    Parameters
    ----------
    coords : ndarray of shape (P, D)
        The coordinates of the considered points.
    spacing : float, optional
        The minimal allowed distance separating points in `coords`. To find the
        maximum number of peaks, use `spacing=1`. See also `p_norm`.
    p_norm : float, optional
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance. See also :func:`numpy.linalg.norm`.
    min_split_size : int or None, optional
        Minimum split size used to process ``coords`` by batch to save
        memory. If None, the memory saving strategy is not applied.
    max_out : int, optional
        If not None, only the first ``max_out`` candidates are returned.
    max_split_size : int, optional
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
    output : ndarray of shape (S, D), same dtype as `coords` and S < P
        A subset of the points in `coords` where a minimum spacing is guaranteed.

    Examples
    --------
    >>> coords = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> _ensure_spacing(coords, spacing=3)
    array([[0, 0],
           [3, 3]])

    # Use _Manhatten/rectilinear distance_
    >>> _ensure_spacing(coords, spacing=3, p_norm=1)
    array([[0, 0],
           [2, 2]])
    """
    return _ensure_spacing(
        coords=coords,
        spacing=spacing,
        p_norm=p_norm,
        min_split_size=min_split_size,
        max_out=max_out,
        max_split_size=max_split_size,
    )


from skimage._doctest_adapters import adapt_doctests  # noqa: E402

adapt_doctests(globals())
