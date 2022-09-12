from ._pnpoly import _grid_points_in_poly, _points_in_poly


def grid_points_in_poly_label(shape, verts):
    """Test whether points on a specified grid are inside a polygon.

    For each ``(r, c)`` coordinate on a grid, i.e. ``(0, 0)``, ``(0, 1)`` etc.,
    check what is the relative location to the polygon. For each pixel, it
    assigns one of these values: O - outside, 1 - inside, 2 - vertex, 3 - edge.

    Parameters
    ----------
    shape : tuple (M, N)
        Shape of the grid.
    verts : (V, 2) array
        Specify the V vertices of the polygon, sorted either clockwise
        or anti-clockwise. The first point may (but does not need to be)
        duplicated.

    See Also
    --------
    points_in_poly
    grid_points_in_poly

    Returns
    -------
    labels: (M, N) ndarray of int
        Labels array, with pixels having a label between 0 and 3.
        The meaning of the values is: O - outside, 1 - inside,
        2 - vertex, 3 - edge.

    """
    return _grid_points_in_poly(shape, verts)


def grid_points_in_poly(shape, verts):
    """Test whether points on a specified grid are inside a polygon.

    For each ``(r, c)`` coordinate on a grid, i.e. ``(0, 0)``, ``(0, 1)`` etc.,
    test whether that point lies inside a polygon.

    Note that this function explicitly includes vertices/edges inside the poly.
    For a better control on this behaviour, use ``grid_points_in_poly_label``.

    Parameters
    ----------
    shape : tuple (M, N)
        Shape of the grid.
    verts : (V, 2) array
        Specify the V vertices of the polygon, sorted either clockwise
        or anti-clockwise. The first point may (but does not need to be)
        duplicated.

    See Also
    --------
    points_in_poly
    grid_points_in_poly_label

    Returns
    -------
    mask : (M, N) ndarray of bool
        True where the grid falls inside the polygon.

    """
    output = _grid_points_in_poly(shape, verts)
    return output.astype(bool)


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
