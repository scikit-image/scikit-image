__all__ = ['grad_func', 'inpaint_point']

import numpy as np
from skimage.morphology import disk

KNOWN = 0
BAND = 1
INSIDE = 2


def grad_func(i, j, flag, array, channel=-1):
    """This function calculates the gradient of the speed/image of a pixel
    depending on the value of the flag of its neighbours. The gradient
    is computed using Central Differences method. If the pixel on either side
    of the central pixel are KNOWN or BAND, then their difference is taken as
    the gradient. However, if either of the pixels is not KNOWN nor BAND,
    then the difference is computed as directly between its negihbour and
    itself.

    This function is used to compute the gradient of intensity value, I and
    also the `u` value.

    Parameters
    ---------
    i, j: index values
        of the pixel whose gradient is to be calculated
    flag: ndarray of unsigned integers
    array: either `image` or `u`
    channel: signed integer
        If channel == -1 then the gradient of `u` is to be calculated
        If channel == 0, 1, 2 then the gradient of `image` is to be calculated

    Returns
    ------
    gradUx: float
        The signed gradient of `image` or `u` in X direction
    gradUy: float
        The signed gradient of `image` or `u` in Y direction

    """

    if channel == 0 or channel == 1 or channel == 2:
        #u = array[:, :, channel]
        u = np.array(array, int)
        # i_nbl = i - 1 + (i == 1)
        # i_nbh = i - 1 - (i == u.shape[0] - 2)
        # j_nbl = j - 1 + (j == 1)
        # j_nbh = j - 1 - (j == u.shape[1] - 2)
        factor = 2.0
    elif channel is -1:
        u = np.array(array, int)
        factor = 0.5
        # i_nbl, i_nbh, j_nbl, j_nbh = i, i, j, j
# TODO: Try dict implementation instead of if...elif

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
    i, j: unsigned integers
        Index whose neighborhood is to be calculated
    size: tuple of integers
        Consists of the shape of the padded input image
    epsilon: unsigned integer

    Returns
    ------
    nb: list of tuples
        List of indexes whose cartesian distance to the input pixel index is
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
    """This function does does the actual inpaining. In order for the inpainted
    region to look natural it is important that the intensity value of a pixel
    be in accordance with the intensity value of its neighbors. It is also
    crucial to pass along the gradient information so that edges and sharp
    changes are preserved in this operation and do not lead to blurring.

    The operation of inpainting is pretty simple and straight forward.
    We iterate over all pixels belonging to the epsilon neighbourhood of the
    pixel to be inpainted and whose `flag` values are KNOWN. We use the fact
    that in discrete case the image gradient can be approximated as:

    `I(p) = I(q) + gradI(q)(p - q)`

    Using this as the basic equation we use a normalized weight function to sum
    over all the pixels satisfying the above condition and compute the above
    value. This provides the intensity value of the pixel to be inpainted.

    Parameters
    ---------
    i, j: index of the pixel to be inpainted
    image: ndarray of unsigned integers
        Padded input image
    flag: ndarray of unsigned integers
    u: ndarray of float
    epsilon: unsigned integer
        To compute the epsilon neighbourhood

    Returns
    ------
    image[i, j]: unsigned integer
        Inpainted intensity value

    References
    ----------
    .. [1] http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    """
    Ia, Jx, Jy, norm = 0, 0, 0, 0
    #If the input image is 3 channel. TODO: support for a single channel
    for color in [0]:
        #Compute the gradient of the u or speed at (i, j)
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

                #Comput the gradient of image intensity at (i_nb, j_nb)
                gradIx, gradIy = grad_func(i_nb, j_nb, flag, image,
                                           channel=color)

                # Ia += weight * image[i_nb - 1 + (i_nb == 1),
                #                      j_nb - 1 + (j_nb == 1)]
                Ia += weight * image[i_nb, j_nb]
                Jx -= weight * gradIx * rx
                Jy -= weight * gradIy * ry
                norm += weight

        sat = (Ia / norm + (Jx + Jy) / (np.sqrt(Jx * Jx + Jy * Jy) + 1.0e-20)
               + 0.5)
#        image[i - 1, j - 1] = int(round(sat))
        image[i, j] = int(round(sat))

#    return image[i - 1, j - 1]
