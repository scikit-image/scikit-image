__all__ = ['polygon_clip', 'polygon_area']

import numpy as np


def polygon_clip(rp, cp, r0, c0, r1, c1):
    """Clip a polygon to the given bounding box.

    Parameters
    ----------
    rp, cp : (N,) ndarray of double
        Row and column coordinates of the polygon. If the first and last
        coordinate are not the same, the polygon will be closed automatically.
    (r0, c0), (r1, c1) : double
        Top-left and bottom-right coordinates of the bounding box.

    Returns
    -------
    r_clipped, c_clipped : (M,) ndarray of double
        Coordinates of clipped polygon. The returned polygon will be close,
        i.e. the first coordinate will be the same as the last coordinate.

    Notes
    -----
    This makes use of the Sutherland-Hodgman algorithm for clipping [1]_.

    References
    ----------
    .. [1]  Wikipedia. Sutherland–Hodgman algorithm
            https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm
    """

    rbox, cbox = _bbox(r0, c0, r1, c1)
    r_clipped, c_clipped = _clip_sutherland_hodgman(rp, cp, rbox, cbox)
    # make sure we return a closed polygon
    # as we did when when
    # matplotlib's functions were called.
    if not (r_clipped[0] == r_clipped[-1] and c_clipped[0] == c_clipped[-1]):
        # cast to list
        r_clipped = list(r_clipped)
        c_clipped = list(c_clipped)
        # Append the first element to close the polygon
        r_clipped.append(r_clipped[0])
        c_clipped.append(c_clipped[0])
    return np.asarray(r_clipped), np.asarray(c_clipped)


def polygon_area(pr, pc):
    """Compute the area of a polygon.

    Parameters
    ----------
    pr, pc : (N,) array of float
        Polygon row and column coordinates.

    Returns
    -------
    a : float
        Area of the polygon.
    """
    pr = np.asarray(pr)
    pc = np.asarray(pc)
    return 0.5 * np.abs(np.sum((pc[:-1] * pr[1:]) - (pc[1:] * pr[:-1])))


def _clip_sutherland_hodgman(rp, cp, rclip, cclip):
    """Clip an arbitrary polygon by an other convex polygon.

    Implements the Sutherland-Hodgman algorithm to clip one polygon by
    an other convex polygon [1]_.

    Parameters
    ----------
    rp, cp : (N,) ndarray of double
        Row and column coordinates of the polygon. Open polygons are OK.
    rclip, cclip : (N,) ndarray of double
        Row and column coordinates of the clipping polygon. This polygon should
        be closed. The polygon is also assumed to be in the clock-wise
        direction.

    Returns
    -------
    r_clipped, c_clipped : (M,) ndarray of double
        Coordinates of clipped polygon. This polygon may potentially be open.

    References
    ----------
    .. [1]  Wikipedia. Sutherland–Hodgman algorithm
            https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm
    """

    def intersect(start, end, normal, point):
        # equation 1
        # n_y y + n_x x - n @ p = 0

        # equation 2
        # dr * y + dc * x - [dr dc] @ s = 0
        dr = end[0] - start[0]
        dc = end[1] - start[1]
        normal_2 = np.asarray([dc, -dr])

        # We now have a system of equations
        # [[normal],     [[y],  =   [[n @ p],
        #  [normal_2]]    [x]]       [n2 @ s]]
        # That we should solve
        A = np.stack([normal, normal_2], axis=0)
        b = np.asarray([normal @ point, normal_2 @ s])
        return np.linalg.solve(A, b)

    # each edge is defined by a normal, and a single point on the edge
    dr = rclip[1:] - rclip[:-1]
    dc = cclip[1:] - cclip[:-1]
    clip_normals = np.stack([dc, -dr], axis=1)
    clip_points = np.stack([rclip[:-1], cclip[:-1]], axis=1)

    out_coords = list(zip(rp, cp))
    # start and end points of the edge
    for normal, point in zip(clip_normals, clip_points):
        coords = out_coords
        out_coords = []
        s = coords[-1]
        # This test is true for a convex, clockwise polygon
        start_in = normal @ s - normal @ point >= 0
        for e in coords:
            end_in = normal @ e - normal @ point >= 0
            if end_in:
                if not start_in:
                    out_coords.append(intersect(s, e, normal, point))
                out_coords.append(e)
            elif start_in:
                out_coords.append(intersect(s, e, normal, point))
            s = e
            start_in = end_in

    return zip(*out_coords)


def _bbox(r0, c0, r1, c1):
    r = np.asarray([r0, r0, r1, r1, r0])
    c = np.asarray([c0, c1, c1, c0, c0])
    return r, c
