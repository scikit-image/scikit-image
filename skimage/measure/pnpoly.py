from ._pnpoly import _grid_points_in_poly, _points_in_poly


def grid_points_in_poly(shape, verts, return_labels=False):
    """Test whether points on a specified grid are inside a polygon.

    For each ``(r, c)`` coordinate on a grid, i.e. ``(0, 0)``, ``(0, 1)`` etc.,
    test whether that point lies inside a polygon.

    Parameters
    ----------
    shape : tuple (M, N)
        Shape of the grid.
    verts : (V, 2) array
        Specify the V vertices of the polygon, sorted either clockwise
        or anti-clockwise. The first point may (but does not need to be)
        duplicated.
    return_labels: bool, optional
        If ``True``, a label mask will also be returned along with the default
        boolean mask. The possible labels are: O - outside, 1 - inside,
        2 - vertex, 3 - edge.

    See Also
    --------
    points_in_poly

    Returns
    -------
    mask : (M, N) ndarray of bool
        True where the grid falls inside the polygon.
    labels: (M, N) ndarray of labels (integers)
        Labels array, with pixels having a label between 0 and 3.
        This is only returned if `return_labels` is set to True.

    """
    return _grid_points_in_poly(shape, verts, return_labels=return_labels)


def points_in_poly(points, verts):
    """Test whether points lie inside a polygon.

    Parameters
    ----------
    points : (N, 2) array
        Input points, ``(x, y)``.
    verts : (M, 2) array
        Vertices of the polygon, sorted either clockwise or anti-clockwise.
        The first point may (but does not need to be) duplicated.

    See Also
    --------
    grid_points_in_poly

    Returns
    -------
    mask : (N,) array of bool
        True if corresponding point is inside the polygon.

    """
    return _points_in_poly(points, verts)
