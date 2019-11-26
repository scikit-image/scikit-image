import numpy as np
from ._draw_nd import _line_nd_cy


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
    
    n_points = np.ceil(np.abs(delta[x_dim])).astype(int)
    if endpoint:
        n_points += 1

    step = np.sign(delta).astype(int)
    mask_not_x = np.ones_like(start, dtype=bool)
    mask_not_x[x_dim] = 0

    error = 2 * delta_abs - delta_abs[x_dim]
    cur = np.round(start).astype(int)
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
