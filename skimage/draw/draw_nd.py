import numpy as np
from ._draw_nd import _line_nd_cy


def _round_safe(coords):
    """Round coords while ensuring successive values are less than 1 apart.

    When rounding coordinates for `line_nd`, we want coordinates that are less
    than 1 apart (always the case, by design) to remain less than one apart.
    However, NumPy rounds values to the nearest *even* integer, so:

    >>> np.round([0.5, 1.5, 2.5, 3.5, 4.5])
    array([0., 2., 2., 4., 4.])

    So, for our application, we detect whether the above case occurs, and use
    ``np.floor`` if so. It is sufficient to detect that the first coordinate
    falls on 0.5 and that the second coordinate is 1.0 apart, since we assume
    by construction that the inter-point distance is less than or equal to 1
    and that all successive points are equidistant.

    Parameters
    ----------
    coords : 1D array of float
        The coordinates array. We assume that all successive values are
        equidistant (``np.all(np.diff(coords) = coords[1] - coords[0])``)
        and that this distance is no more than 1
        (``np.abs(coords[1] - coords[0]) <= 1``).

    Returns
    -------
    rounded : 1D array of int
        The array correctly rounded for an indexing operation, such that no
        successive indices will be more than 1 apart.

    Examples
    --------
    >>> coords0 = np.array([0.5, 1.25, 2., 2.75, 3.5])
    >>> _round_safe(coords0)
    array([0, 1, 2, 3, 4])
    >>> coords1 = np.arange(0.5, 8, 1)
    >>> coords1
    array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    >>> _round_safe(coords1)
    array([0, 1, 2, 3, 4, 5, 6, 7])
    """
    if (len(coords) > 1
            and coords[0] % 1 == 0.5
            and coords[1] - coords[0] == 1):
        _round_function = np.floor
    else:
        _round_function = np.round
    return _round_function(coords).astype(int)


def line_nd(start, stop, *, endpoint=False, integer=True):
    """Draw a single-pixel thick line in n dimensions.

    The line produced will be ndim-connected. That is, two subsequent
    pixels in the line will be either direct or diagonal neighbors in
    n dimensions.

    Parameters
    ----------
    start : array-like, shape (N,)
        The start coordinates of the line.
    stop : array-like, shape (N,)
        The end coordinates of the line.
    endpoint : bool, optional
        Whether to include the endpoint in the returned line. Defaults
        to False, which allows for easy drawing of multi-point paths.
    integer : bool, optional
        Whether to round the coordinates to integer. If True (default),
        the returned coordinates can be used to directly index into an
        array. `False` could be used for e.g. vector drawing.

    Returns
    -------
    coords : tuple of arrays
        The coordinates of points on the line.

    Examples
    --------
    >>> lin = line_nd((1, 1), (5, 2.5), endpoint=False)
    >>> lin
    (array([1, 2, 3, 4]), array([1, 1, 2, 2]))
    >>> im = np.zeros((6, 5), dtype=int)
    >>> im[lin] = 1
    >>> im
    array([[0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> line_nd([2, 1, 1], [5, 5, 2.5], endpoint=True)
    (array([2, 3, 4, 4, 5]), array([1, 2, 3, 4, 5]), array([1, 1, 2, 2, 2]))
    """
    start = np.asarray(start)
    stop = np.asarray(stop)
    npoints = int(np.ceil(np.max(np.abs(stop - start))))
    if endpoint:
        npoints += 1

    coords = np.linspace(start, stop, num=npoints, endpoint=endpoint).T
    if integer:
        for dim in range(len(start)):
            coords[dim, :] = _round_safe(coords[dim, :])

        coords = coords.astype(int)

    return tuple(coords)


def _line_nd_py(cur, stop, step, *, delta, x_dim, endpoint, error, coords):

    ndim = error.shape[0]
    n_points = coords.shape[1]

    for i_pt in range(n_points):
        for i_dim in range(ndim):
            coords[i_dim, i_pt] = cur[i_dim]
            if error[i_dim] > 0:
                coords[i_dim, i_pt] += step[i_dim]
                error[i_dim] -= 2 * delta[x_dim]
            error[i_dim] += 2 * delta[i_dim]
            cur[i_dim] = coords[i_dim, i_pt]


def my_line_nd(start, stop, *, endpoint=False, integer=True):
    """Draw a single-pixel thick line in n dimensions.

    The line produced will be ndim-connected. That is, two subsequent
    pixels in the line will be either direct or diagonal neighbours in
    n dimensions.

    Parameters
    ----------
    start : array-like, shape (N,)
        The start coordinates of the line.
    stop : array-like, shape (N,)
        The end coordinates of the line.
    endpoint : bool, optional
        Whether to include the endpoint in the returned line. Defaults
        to False, which allows for easy drawing of multi-point paths.
    integer : bool, optional
        Whether to round the coordinates to integer. If True (default),
        the returned coordinates can be used to directly index into an
        array. `False` could be used for e.g. vector drawing.

    Returns
-------
    coords : tuple of arrays
        The coordinates of points on the line.

    Examples
    --------
    >>> lin = line_nd((1, 1), (5, 2.5), endpoint=False)
    >>> lin
    (array([1, 2, 3, 4]), array([1, 1, 2, 2]))
    >>> im = np.zeros((6, 5), dtype=int)
    >>> im[lin] = 1
    >>> im
    array([[0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 1, 0, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0]])
    >>> line_nd([2, 1, 1], [5, 5, 2.5], endpoint=True)
    (array([2, 3, 4, 4, 5]), array([1, 2, 3, 4, 5]), array([1, 1, 2, 2, 2]))
    """
    start = np.asarray(start)
    stop = np.asarray(stop)
    delta = stop - start
    delta_abs = np.abs(delta)
    x_dim = np.argmax(delta_abs)

    m = (stop - start) / (stop[x_dim] - start[x_dim])
    q = (stop[x_dim] * start - start[x_dim] * stop) / delta[x_dim]

    start_int = np.floor(start + 0.5).astype(int)
    stop_int = np.floor(stop + 0.5).astype(int)
    delta_int = stop_int - start_int
    delta_abs_int = np.abs(delta_int)
    
    n_points = abs(stop_int[x_dim] - start_int[x_dim])
    if endpoint:
        n_points += 1

    step = np.sign(delta).astype(int)

    error = (2 * (m * start_int[x_dim] + q - start_int) - 1) * delta_abs_int[x_dim]
    error[x_dim] = 0
    error_int = error.astype(int)
    coords = np.zeros([len(start), n_points], dtype=np.intp)

    _line_nd_cy(start_int, step,
                delta=delta_abs_int, x_dim=x_dim,
                error=error_int, coords=coords)

    return tuple(coords)


def _line_nd(cur, step, delta, x_dim, error, coords):

    n_dim = error.shape[0]
    n_points = coords.shape[1]

    for i_pt in range(n_points):
        for i_dim in range(n_dim):
            coords[i_dim, i_pt] = cur[i_dim]
            if error[i_dim] > 0:
                coords[i_dim, i_pt] += step[i_dim]
                error[i_dim] -= 2 * delta[x_dim]
            error[i_dim] += 2 * delta[i_dim]
            cur[i_dim] = coords[i_dim, i_pt]


tests = [
    ((0, 0), (2, 2)), # m=1 (x0 is even)
    ((1, 1), (3, 3)), # m=1 (x0 is odd)
    ((0, 0), (4, 2)), # m=2 (x0 is even)
    ((1, 1), (5, 3)), # m=2 (x0 is odd)
    ((1, 1), (3, 5)), # x is the second coordinate
    ((1, 1), (5, 2.5)), # y1 is decimal
    ((1.1, 1.4), (5.3, 2.2)), # all decimals (this produces a bug - a repeated point in the original implementation, due to the call to np.ceil (maybe a -0.5 is due in there))
    ((0, 0), (5, -3)), # m<0
    ((0, 0), (3, -5)), # m<0, x is the second coordinate
    ((2, 1, 1), (5, 5, 2.5)), # 3 dimensions
    ((2, 1, 1), (5, 5, -2.5)), # 3 dimensions, m>0 and m<0
    ((2, 1, 0), (5, 5, -1.5)), # 3 dimensions, m>0 and m<0, shifting causes different shape in the old implementation (compared to previous test)
    ((2, 1, 1), (5, 5, -2.4)), # just need to tease out why the results are what they are in this and the next example
    ((2, 1, 1), (5, 5, -2.6))

]

for start, stop in tests:
    print(start, stop)
    print(line_nd(start, stop, endpoint=True))
    print(my_line_nd(start, stop, endpoint=True))
    print()