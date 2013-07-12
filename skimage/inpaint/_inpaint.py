import numpy as np
from skimage.morphology import disk


__all__ = ['grad_func', 'ep_neighbor', 'inpaint_point']


KNOWN = 0
BAND = 1
INSIDE = 2


def grad_func(i, j, flag, array, channel=-1):
    """This function calculates the gradient of the distance/image of a pixel
    depending on the value of the flag of its neighbours. The gradient
    is computed using Central Differences method.

    This function is used to compute the gradient of intensity value, I and
    also the `u` value.

    Parameters
    ---------
    i, j : int
        Row and column index value of the pixel whose gradient is to be
        calculated
    flag : array
        Array marking pixels as known, along the boundary to be solved, or
        inside the unknown region: 0 = KNOWN, 1 = BAND, 2 = INSIDE
    array : array
        Either `image` or `u`
    channel : integer
        If channel == -1 then the gradient of `u` is to be calculated
        If channel == 0 then the gradient of `image` is to be calculated

    Returns
    ------
    gradUx : float
        The signed gradient of `image` or `u` in X direction
    gradUy : float
        The signed gradient of `image` or `u` in Y direction

    """

    if channel == 0:
        u = np.array(array, int)
        factor = 2.0
    elif channel is -1:
        u = np.array(array, float)
        factor = 0.5

    if flag[i, j + 1] != INSIDE and flag[i, j - 1] != INSIDE:
        gradUx = np.subtract(u[i, j + 1], u[i, j - 1]) * factor
    elif flag[i, j + 1] != INSIDE and flag[i, j - 1] == INSIDE:
        gradUx = np.subtract(u[i, j + 1], u[i, j])
    elif flag[i, j + 1] == INSIDE and flag[i, j - 1] != INSIDE:
        gradUx = np.subtract(u[i, j], u[i, j - 1])
    elif flag[i, j + 1] == INSIDE and flag[i, j - 1] == INSIDE:
        gradUx = 0

    if flag[i + 1, j] != INSIDE and flag[i - 1, j] != INSIDE:
        gradUy = np.subtract(u[i + 1, j], u[i - 1, j]) * factor
    elif flag[i + 1, j] != INSIDE and flag[i - 1, j] == INSIDE:
        gradUy = np.subtract(u[i + 1, j], u[i, j])
    elif flag[i + 1, j] == INSIDE and flag[i - 1, j] != INSIDE:
        gradUy = np.subtract(u[i, j], u[i - 1, j])
    elif flag[i + 1, j] == INSIDE and flag[i - 1, j] == INSIDE:
        gradUy = 0

    # if flag[i, j + 1] != INSIDE and flag[i, j - 1] != INSIDE:
    #     gradUx = np.subtract(u[i_nbl, j_nbh + 1], u[i_nbl, j_nbl - 1]) * factor
    # elif flag[i, j + 1] != INSIDE and flag[i, j - 1] == INSIDE:
    #     gradUx = np.subtract(u[i_nbl, j_nbh + 1], u[i_nbl, j_nbl])
    # elif flag[i, j + 1] == INSIDE and flag[i, j - 1] != INSIDE:
    #     gradUx = np.subtract(u[i_nbl, j_nbh], u[i_nbl, j_nbl - 1])
    # elif flag[i, j + 1] == INSIDE and flag[i, j - 1] == INSIDE:
    #     gradUx = 0

    # if flag[i + 1, j] != INSIDE and flag[i - 1, j] != INSIDE:
    #     gradUy = np.subtract(u[i_nbh + 1, j_nbl], u[i_nbl - 1, j_nbl]) * factor
    # elif flag[i + 1, j] != INSIDE and flag[i - 1, j] == INSIDE:
    #     gradUy = np.subtract(u[i_nbh + 1, j_nbl], u[i_nbl, j_nbl])
    # elif flag[i + 1, j] == INSIDE and flag[i - 1, j] != INSIDE:
    #     gradUy = np.subtract(u[i_nbh, j_nbl], u[i_nbl - 1, j_nbl])
    # elif flag[i + 1, j] == INSIDE and flag[i - 1, j] == INSIDE:
    #     gradUy = 0

    return gradUx, gradUy


def ep_neighbor(i, j, size, epsilon):
    """This computes the epsilon neighbourhood of the `(i, j)` pixel.

    Parameters
    ---------
    i, j : int
        Row and column index value of the pixel whose neighbourhood
        is to be calculated
    size : tuple of integers
        Shape of the padded input image
    epsilon : integer
        Neighbourhood of (i, j) to be considered for inpainting

    Returns
    ------
    nb : list of tuples
        List of indices whose cartesian distance to the input pixel index is
        less than epsilon
    """
    nb = []
    indices = np.transpose(np.where(disk(epsilon)))
    center_ind = indices - [epsilon, epsilon] + [i, j]
    for ind in center_ind:
        if (ind >= [0, 0]).all() and (ind <= np.array(size) - [1, 1]).all():
            nb.append(ind)

    return nb
