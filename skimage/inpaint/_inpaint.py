__all__ = ['grad_func', 'inpaint_point']

import numpy as np

KNOWN = 0
BAND = 1
INSIDE = 2


def grad_func(i, j, flag, array, channel=-1):
    """This function calculates the gradient of the speed/image of a pixel
    depending on the value of the flag of its neighbours. The gradient
    is computed using Central Differences.
    """

    if channel == 0 or channel == 1 or channel == 2:
        #u = array[:, :, channel]
        u = np.array(array, int)
        i_nbl = i - 1 + (i == 1)
        i_nbh = i - 1 - (i == u.shape[0] - 2)
        j_nbl = j - 1 + (j == 1)
        j_nbh = j - 1 - (j == u.shape[1] - 2)
        factor = 2.0
    elif channel is -1:
        u = np.array(array, int)
        factor = 0.5
        i_nbl, i_nbh, j_nbl, j_nbh = i, i, j, j
# TODO: Try dict implementation instead of if...elif
    if flag[i, j + 1] != INSIDE and flag[i, j - 1] != INSIDE:
        gradUx = np.subtract(u[i_nbl, j_nbh + 1], u[i_nbl, j_nbl - 1]) * factor
    elif flag[i, j + 1] != INSIDE and flag[i, j - 1] == INSIDE:
        gradUx = np.subtract(u[i_nbl, j_nbh + 1], u[i_nbl, j_nbl])
    elif flag[i, j + 1] == INSIDE and flag[i, j - 1] != INSIDE:
        gradUx = np.subtract(u[i_nbl, j_nbh], u[i_nbl, j_nbl - 1])
    elif flag[i, j + 1] == INSIDE and flag[i, j - 1] == INSIDE:
        gradUx = 0

    if flag[i + 1, j] != INSIDE and flag[i - 1, j] != INSIDE:
        gradUy = np.subtract(u[i_nbh + 1, j_nbl], u[i_nbl - 1, j_nbl]) * factor
    elif flag[i + 1, j] != INSIDE and flag[i - 1, j] == INSIDE:
        gradUy = np.subtract(u[i_nbh + 1, j_nbl], u[i_nbl, j_nbl])
    elif flag[i + 1, j] == INSIDE and flag[i - 1, j] != INSIDE:
        gradUy = np.subtract(u[i_nbh, j_nbl], u[i_nbl - 1, j_nbl])
    elif flag[i + 1, j] == INSIDE and flag[i - 1, j] == INSIDE:
        gradUy = 0

    return gradUx, gradUy


def inpaint_point(i, j, image, flag, u, epsilon):
    Ia, Jx, Jy, norm = 0, 0, 0, 0
    #If the input image is 3 channel. TODO: support for a single channel
    for color in [0]:
        #Compute the gradient of the u or speed at (i, j)
        gradUx, gradUy = grad_func(i, j, flag, u, channel=-1)
        for i_nb in xrange(i - epsilon, i + epsilon):
            for j_nb in xrange(j - epsilon, j + epsilon):
                if (i_nb > 0 and j_nb > 0 and i_nb < (u.shape[0] - 1)
                        and j_nb < (u.shape[1] - 1)):
                    cart_d = (j_nb - j) * (j_nb - j) + (i_nb - i) * (i_nb - i)
                    if (flag[i_nb, j_nb] == KNOWN and
                            cart_d <= epsilon ** 2):
                        # gradUx, gradUy = grad_func(i_nb, j_nb, flag, u,
                        #                            channel=-1)
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

                        Ia += weight * image[i_nb - 1 + (i_nb == 1),
                                             j_nb - 1 + (j_nb == 1)]
                        Jx -= weight * gradIx * rx
                        Jy -= weight * gradIy * ry
                        norm += weight

        sat = (Ia / norm + (Jx + Jy) / (np.sqrt(Jx * Jx + Jy * Jy) + 1.0e-20)
               + 0.5)
#        image[i - 1, j - 1] = int(round(sat))
        image[i, j] = int(round(sat))

#    return image[i - 1, j - 1]
