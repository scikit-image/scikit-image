import numpy as np

"""
Code copied from https://github.com/machine-shop/deepwings/blob/master/deepwings/method_features_extraction/image_processing.py#L185
"""


def moore_neighborhood(current, backtrack):  # y, x
    """Returns clockwise list of pixels from the moore neighborhood of current\
    pixel:
    The first element is the coordinates of the backtrack pixel.
    The following elements are the coordinates of the neighboring pixels in
    clockwise order.
    Parameters
    ----------
    current ([y, x]): Coordinates of the current pixel
    backtrack ([y, x]): Coordinates of the backtrack pixel
    Returns
    -------
    List of coordinates of the moore neighborood pixels, or 0 if the backtrack
    pixel is not a current pixel neighbor
    """

    operations = np.array(
        [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
    )
    neighbors = (current + operations).astype(int)

    for i, point in enumerate(neighbors):
        if np.all(point == backtrack):
            # we return the sorted neighborhood
            return np.concatenate((neighbors[i:], neighbors[:i]))
    return 0  # TODO: FIX. Should not return 0 because it will break boundary


# TODO: work with the coordinates, to make it more generic
# Any array of coordinates should work
def boundary_tracing(region):
    """Coordinates of the region's boundary. The region must not have isolated
    points.
    Parameters
    ----------
    region : obj
        Obtained with skimage.measure.regionprops()
    Returns
    -------
    boundary : 2D array
        List of coordinates of pixels in the boundary
        The first element is the most upper left pixel of the region.
        The following coordinates are in clockwise order.
    """

    # creating the binary image
    coords = region.coords
    maxs = np.amax(coords, axis=0)

    # should consider the minimum as well
    # we want a image of minimal size
    binary = np.zeros((maxs[0] + 2, maxs[1] + 2))

    x = coords[:, 1]
    y = coords[:, 0]
    binary[tuple([y, x])] = 1

    # initilization
    # starting point is the most upper left point
    idx_start = 0
    while True:  # asserting that the starting point is not isolated
        start = [y[idx_start], x[idx_start]]
        focus_start = binary[start[0] - 1 : start[0] + 2, start[1] - 1 : start[1] + 2]
        if np.sum(focus_start) > 1:
            break
        idx_start += 1

    # Determining backtrack pixel for the first element
    if binary[start[0] + 1, start[1]] == 0 and binary[start[0] + 1, start[1] - 1] == 0:
        backtrack_start = [start[0] + 1, start[1]]
    else:
        backtrack_start = [start[0], start[1] - 1]

    current = start
    backtrack = backtrack_start
    boundary = []
    counter = 0

    while True:
        neighbors_current = moore_neighborhood(current, backtrack)
        y = neighbors_current[:, 0]
        x = neighbors_current[:, 1]
        idx = np.argmax(binary[tuple([y, x])])
        boundary.append(current)
        backtrack = neighbors_current[idx - 1]
        current = neighbors_current[idx]
        counter += 1

        if np.all(current == start) and np.all(backtrack == backtrack_start):
            break

    return np.array(boundary)
