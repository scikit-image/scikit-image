import numpy as np
from scipy import signal
import pdb


def approximate_polygon(coords, tolerance):
    """Approximate a polygonal chain with the specified tolerance.

    It is based on the Douglas-Peucker algorithm.

    Note that the approximated polygon is always within the convex hull of the
    original polygon.

    Parameters
    ----------
    coords : (N, 2) array
        Coordinate array.
    tolerance : float
        Maximum distance from original points of polygon to approximated
        polygonal chain. If tolerance is 0, the original coordinate array
        is returned.

    Returns
    -------
    coords : (M, 2) array
        Approximated polygonal chain where M <= N.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Ramer-Douglas-Peucker_algorithm
    """
    """
    if tolerance <= 0:
        return coords

    if coords.shape[0] <= 2:
        return coords

    chain = np.zeros(coords.shape[0], 'bool')
    chain[0] = True
    chain[coords.shape[0] - 1] = True
    pos_stack = [(0, coords.shape[0] - 1)]
    while len(pos_stack) != 0:
        start, end = pos_stack.pop()
        index, dmax = _max_perp_dist(coords, start, end)
        if dmax > tolerance:
            pos_stack.append((start, index))
            pos_stack.append((index, end))
            chain[index] = True

    return coords[chain, :]
    """
    if tolerance <= 0:
        return coords

    chain = np.zeros(coords.shape[0], 'bool')
    # pre-allocate distance array for all points
    dists = np.zeros(coords.shape[0])
    chain[0] = True
    chain[-1] = True
    pos_stack = [(0, chain.shape[0] - 1)]
    end_of_chain = False

    while not end_of_chain:
        start, end = pos_stack.pop()
        # determine properties of current line segment
        r0, c0 = coords[start, :]
        r1, c1 = coords[end, :]
        dr = r1 - r0
        dc = c1 - c0
        segment_angle = - np.arctan2(dr, dc)
        segment_dist = c0 * np.sin(segment_angle) + r0 * np.cos(segment_angle)

        # select points in-between line segment
        segment_coords = coords[start + 1:end, :]
        segment_dists = dists[start + 1:end]

        # check whether to take perpendicular or euclidean distance with
        # inner product of vectors

        # vectors from points -> start and end
        dr0 = segment_coords[:, 0] - r0
        dc0 = segment_coords[:, 1] - c0
        dr1 = segment_coords[:, 0] - r1
        dc1 = segment_coords[:, 1] - c1
        # vectors points -> start and end projected on start -> end vector
        projected_lengths0 = dr0 * dr + dc0 * dc
        projected_lengths1 = - dr1 * dr - dc1 * dc
        perp = np.logical_and(projected_lengths0 > 0,
                              projected_lengths1 > 0)
        eucl = np.logical_not(perp)
        segment_dists[perp] = np.abs(
            (segment_coords[perp, 0] * dc - segment_coords[perp, 1] * dr \
            + (r1 * c0) - (c1 * r0)) \
            / float(((dr ** 2 + dc ** 2) ** 0.5))
        )
        segment_dists[eucl] = np.minimum(
            # distance to start point
            np.sqrt(dc0[eucl] ** 2 + dr0[eucl] ** 2),
            # distance to end point
            np.sqrt(dc1[eucl] ** 2 + dr1[eucl] ** 2)
        )

        if np.any(segment_dists > tolerance):
            # select point with maximum distance to line
            new_end = start + np.argmax(segment_dists) + 1
            pos_stack.append((new_end, end))
            pos_stack.append((start, new_end))
            chain[new_end] = True

        if len(pos_stack) == 0:
            end_of_chain = True

    return coords[chain, :]


def _max_perp_dist(coords, start, end):
    """Helper function for approximate_polygon.

    For each point in COORDS, it calculates the perpendicular distance from the
    line connecting START and END.

    It returns the index and the distance of the point that has the maximum
    distance.

    Parameters
    ----------
    coords : (N, 2) array
        Coordinate array.

    Returns
    -------
    index : integer
        Index of the point with maximum distance.
    dmax : float
        The maximum perpendicular distance.
    """
    dmax = 0
    index = 0
    p1_x, p1_y = coords[start, :]
    p2_x, p2_y = coords[end, :]

    for i in range(start + 1, end):
        x, y = coords[i, :]
        perp_dist = abs(float((p2_y - p1_y) * x - (p2_x - p1_x) * y + \
                        (p2_x * p1_y) - (p2_y * p1_x))) / \
                        (((p2_y - p1_y) ** 2 + (p2_x - p1_x) ** 2) ** 0.5)
        if perp_dist > dmax:
            index = i
            dmax = perp_dist

    return index, dmax

# B-Spline subdivision
_SUBDIVISION_MASKS = {
    # degree: (mask_even, mask_odd)
    #         extracted from (degree + 2)th row of Pascal's triangle
    1: ([1, 1], [1, 1]),
    2: ([3, 1], [1, 3]),
    3: ([1, 6, 1], [0, 4, 4]),
    4: ([5, 10, 1], [1, 10, 5]),
    5: ([1, 15, 15, 1], [0, 6, 20, 6]),
    6: ([7, 35, 21, 1], [1, 21, 35, 7]),
    7: ([1, 28, 70, 28, 1], [0, 8, 56, 56, 8]),
}


def subdivide_polygon(coords, degree=2, preserve_ends=False):
    """Subdivision of polygonal curves using B-Splines.

    Note that the resulting curve is always within the convex hull of the
    original polygon. Circular polygons stay closed after subdivision.

    Parameters
    ----------
    coords : (N, 2) array
        Coordinate array.
    degree : {1, 2, 3, 4, 5, 6, 7}, optional
        Degree of B-Spline. Default is 2.
    preserve_ends : bool, optional
        Preserve first and last coordinate of non-circular polygon. Default is
        False.

    Returns
    -------
    coords : (M, 2) array
        Subdivided coordinate array.

    References
    ----------
    .. [1] http://mrl.nyu.edu/publications/subdiv-course2000/coursenotes00.pdf
    """
    if degree not in _SUBDIVISION_MASKS:
        raise ValueError("Invalid B-Spline degree. Only degree 1 - 7 is "
                         "supported.")

    circular = np.all(coords[0, :] == coords[-1, :])

    method = 'valid'
    if circular:
        # remove last coordinate because of wrapping
        coords = coords[:-1, :]
        # circular convolution by wrapping boundaries
        method = 'same'

    mask_even, mask_odd = _SUBDIVISION_MASKS[degree]
    # divide by total weight
    mask_even = np.array(mask_even, np.float) / (2 ** degree)
    mask_odd = np.array(mask_odd, np.float) / (2 ** degree)

    even = signal.convolve2d(coords.T, np.atleast_2d(mask_even), mode=method,
                             boundary='wrap')
    odd = signal.convolve2d(coords.T, np.atleast_2d(mask_odd), mode=method,
                            boundary='wrap')

    out = np.zeros((even.shape[1] + odd.shape[1], 2))
    out[1::2] = even.T
    out[::2] = odd.T

    if circular:
        # close polygon
        out = np.vstack([out, out[0, :]])

    if preserve_ends and not circular:
        out = np.vstack([coords[0, :], out, coords[-1, :]])

    return out
