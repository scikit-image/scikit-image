import numpy as np

"""
Code copied from https://github.com/machine-shop/deepwings/blob/master/deepwings/method_features_extraction/image_processing.py#L185
"""


def _moore_neighborhood(current, backtrack):  # y, x
    """Returns clockwise list of pixels from the moore neighborhood of current
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
    else:
        raise RuntimeError("The backtrack is not on the neighborhood")


def trace_boundary(coords):
    """Coordinates of the region's boundary. The region must not have isolated
    points.
    
    Parameters
    ----------
    coords : obj
        Obtained with skimage.measure.regionprops()
        
    Returns
    -------
    boundary : 2D array
        List of coordinates of pixels in the boundary
        The first element is the most upper left pixel of the region.
        The following coordinates are in clockwise order.
    """

    def starting_point():
        """
        Returns where we will start the search for the boundary. Current it
        returns (max_x, max_y). That is, the basic example:

            [0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0],

        Will start in the (4, 4) point because it has a region limit at
        (1, 4) and another at (4, 3). On a t-shaped example, where the region
        has borders in very far away points, say (1, 20) and (20, 1), I fear
        that this method will begin its search at (20, 20), a point very
        far from the boundary
        """
        return np.amax(coords, axis=0)

    maxs = starting_point()

    # should consider the minimum as well
    # we want a image of minimal size
    binary = np.zeros((maxs[0] + 2, maxs[1] + 2))

    x = coords[:, 1]
    y = coords[:, 0]
    # Sets the original region as 1s
    binary[tuple([y, x])] = 1

    def find_boundary_start():

        # initilization
        # starting point is the most upper left point
        idx_start = 0
        while True:  # asserting that the starting point is not isolated
            start = [y[idx_start], x[idx_start]]

            # Focus Start is a 3x3 matrix centered around start
            focus_start = binary[start[0] - 1 : start[0] + 2, start[1] - 1 : start[1] + 2]

            # If the current pixel is not the only one set to one in its
            # 3x3 region, we stop
            if np.sum(focus_start) > 1:
                break
            idx_start += 1

        # Determining backtrack pixel for the first element
        if binary[start[0] + 1, start[1]] == 0 and binary[start[0] + 1, start[1] - 1] == 0:
            backtrack_start = [start[0] + 1, start[1]]
        else:
            backtrack_start = [start[0], start[1] - 1]
        
        return backtrack_start, start

    backtrack_start, start = find_boundary_start()
    current = start
    backtrack = backtrack_start
    boundary = []
    counter = 0

    while True:
        neighbors_ids = _moore_neighborhood(current, backtrack)
        y = neighbors_ids[:, 0]
        x = neighbors_ids[:, 1]
        neighbors = binary[tuple([y, x])]
        idx = np.argmax(neighbors)
        boundary.append(current)
        backtrack = neighbors_ids[idx - 1]
        current = neighbors_ids[idx]
        counter += 1

        if np.all(current == start) and np.all(backtrack == backtrack_start):
            break

    return np.array(boundary)


def trace_boundary_new(coords):
    pass

"""
...
current = find_boundary_start(...)
offsets = [[-1, 0], [-1, 1], [0, 1], [1, 1], [1, 0], [1, -1], [0, -1], [-1, -1]]
pointer = 0  # ensure that pointer is never larger than len(offsets)
while True:
    neighbor = current + offsets[pointer] # current is in the boundary
    ...
    check_if_boundary():
        save_boundary()
    ...
    stop_criterion()
    ....
    pointer += 1
    pointer %= len(offsets)


[0, 0, 0, 0, 0, 0],
[0, 1, 1, 1, 1, 0],
[0, 1, 1, 1, 1, 0],
[0, 0, 1, 1, 1, 0],
[0, 0, 0, 1, 0, 0],

"""


