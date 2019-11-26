import numpy as np
from ._draw_nd import _line_nd_cy

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

def line_nd(start, stop, *, endpoint=False, integer=True):
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

    q = (stop[x_dim] * start - start[x_dim] * stop) / delta[x_dim]

    start_int = np.round(start).astype(int)
    stop_int = np.round(stop).astype(int)
    
    n_points = abs(stop_int[x_dim] - start_int[x_dim])
    if endpoint:
        n_points += 1

    step = np.sign(delta).astype(int)

    error = 2 * delta_abs - delta_abs[x_dim] + q * 2 * delta_abs[x_dim]
    error[x_dim] = 0
    cur = start_int
    coords = np.zeros([len(start), n_points], dtype=np.intp)

    _line_nd_cy(cur, np.round(stop).astype(int), step, endpoint=endpoint,
                delta=np.round(delta_abs).astype(int), x_dim=x_dim,
                error=np.round(error).astype(int), coords=coords)

    return tuple(coords)


if __name__ == '__main__':
    def test(draw_nd_fn):
        lin = draw_nd_fn((1, 1), (5, 2.5), endpoint=False)
        print(lin)
        print((np.array([1, 2, 3, 4]), np.array([1, 1, 2, 2])))
        im = np.zeros((6, 5), dtype=int)
        im[lin] = 1
        print(im)
        print(np.array([[0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0]]))
        print(draw_nd_fn([2, 1, 1], [5, 5, 2.5], endpoint=True))
        print((np.array([2, 3, 4, 4, 5]), np.array([1, 2, 3, 4, 5]), np.array([1, 1, 2, 2, 2])))

    from _draw_nd import my_line_nd_cython
    test(line_nd)
    test(my_line_nd)
    test(my_line_nd_cython)
