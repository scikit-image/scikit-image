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


def inpaint_point(i, j, image, flag, u, epsilon):
    """This function performs the actual inpainting operation. Inpainting
    involves "filling in" color in regions with unkown intensity values using
    the intensity and gradient information of surrounding known region.

    For further implementation details, see Section 2.3 and Figure 5 in [1]_

    Parameters
    ---------
    i, j : int
        Row and column index value of the pixel to be Inpainted
    image : array
        Padded single channel input image
    flag : array
        Array marking pixels as known, along the boundary to be solved, or
        inside the unknown region: 0 = KNOWN, 1 = BAND, 2 = INSIDE
    u : array
            The distance/time map from the boundary to each pixel.
    epsilon : integer
            Neighbourhood of (i, j) to be considered for inpainting

    Returns
    ------
    image[i, j] : integer
        Inpainted intensity value

    References
    ----------
    .. [1] Telea, A., "An Image Inpainting Technique based on the Fast Marching
            Method", Journal of Graphic Tools (2004).
            http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    """
    Ia, Jx, Jy, norm = 0, 0, 0, 0
    gradUx, gradUy = grad_func(i, j, flag, u, channel=-1)
    nb = ep_neighbor(i, j, image.shape, epsilon)

    for [i_nb, j_nb] in nb:
        if flag[i_nb, j_nb] == KNOWN:
            rx = i - i_nb
            ry = j - j_nb

            dst = 1. / ((rx * rx + ry * ry) *
                        np.sqrt((rx * rx + ry * ry)))
            lev = 1. / (1 + abs(u[i_nb, j_nb] - u[i, j]))
            dirc = rx * gradUx + ry * gradUy

            if abs(dirc) <= 0.01:
                dirc = 1.0e-6
            weight = abs(dst * lev * dirc)

            gradIx, gradIy = grad_func(i_nb, j_nb, flag, image,
                                       channel=0)

            Ia += weight * image[i_nb, j_nb]
            Jx -= weight * gradIx * rx
            Jy -= weight * gradIy * ry
            norm += weight

    sat = (Ia / norm + (Jx + Jy) / (np.sqrt(Jx * Jx + Jy * Jy) + 1.0e-20)
           + 0.5)
    image[i, j] = int(round(sat))
