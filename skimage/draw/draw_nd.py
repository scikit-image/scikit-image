import numpy as np


def line_nd(start, stop, endpoint=False, round=True):
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
    round : bool, optional
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
    [array([1, 2, 3, 4]), array([1, 1, 2, 2])]
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
    [array([2, 3, 4, 4, 5]), array([1, 2, 3, 4, 5]), array([1, 1, 2, 2, 2])]
    """
    start = np.asarray(start)
    stop = np.asarray(stop)
    npoints = np.max(np.abs(stop - start))
    if endpoint:
        npoints += 1
    npoints = int(np.ceil(npoints))
    coords = []
    for dim in range(len(start)):
        dimcoords = np.linspace(start[dim], stop[dim], npoints, endpoint)
        if round:
            dimcoords = np.round(dimcoords).astype(int)
        coords.append(dimcoords)
    return coords
