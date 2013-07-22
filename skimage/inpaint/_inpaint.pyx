#

cimport numpy as cnp
import numpy as np
from libc.math cimport sqrt
from skimage.morphology import disk


__all__ = ['grad_func', 'ep_neighbor', 'inpaint_point']


cdef:
    cnp.uint8_t KNOWN = 0
    cnp.uint8_t BAND = 1
    cnp.uint8_t INSIDE = 2


cdef cnp.float_t[:] grad_func(Py_ssize_t i, Py_ssize_t j, cnp.uint8_t[:, ::1] flag_view, cnp.float_t[:, ::1] array_view, Py_ssize_t channel=1):
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
    flag_view : array
        Array marking pixels as known, along the boundary to be solved, or
        inside the unknown region: 0 = KNOWN, 1 = BAND, 2 = INSIDE
    array_view : array
        Either `image_view` or `u_view`
    channel : integer
        If channel == 1 then the gradient of `u` is calculated
        If channel == 0 then the gradient of `image` is calculated

    Returns
    ------
    gradUx : float
        The signed gradient of `image` or `u` in X direction
    gradUy : float
        The signed gradient of `image` or `u` in Y direction

    """

    cdef:
        cnp.float_t factor
        cnp.float_t[:] gradU = np.zeros(2, dtype=np.float)

    if channel == 0:
        factor = 2.0
    elif channel == 1:
        factor = 0.5

    if flag_view[i, j + 1] != INSIDE:
        if flag_view[i, j - 1] != INSIDE:
            gradU[0] = (array_view[i, j + 1] - array_view[i, j - 1]) * factor
        else:
            gradU[0] = (array_view[i, j + 1] - array_view[i, j])
    else:
        if flag_view[i, j - 1] != INSIDE:
            gradU[0] = (array_view[i, j] - array_view[i, j - 1])
        else:
            gradU[0] = 0

    if flag_view[i + 1, j] != INSIDE:
        if flag_view[i - 1, j] != INSIDE:
            gradU[1] = (array_view[i + 1, j] - array_view[i - 1, j]) * factor
        else:
            gradU[1] = (array_view[i + 1, j] - array_view[i, j])
    else:
        if flag_view[i - 1, j] != INSIDE:
            gradU[1] = (array_view[i, j] - array_view[i - 1, j])
        else:
            gradU[1] = 0

    return gradU


cpdef inpaint_point(Py_ssize_t i, Py_ssize_t j, image, flag, u, epsilon):
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

    cdef:
        cnp.uint8_t[:, ::1] image_view = image
        cnp.uint8_t[:, ::1] flag_view = flag
        cnp.float_t[:, ::1] u_view = u
        cnp.uint8_t[:, ::1] nb_view
        cnp.int8_t[:, ::1] center_ind_view
        cnp.int8_t i_nb, j_nb
        cnp.int8_t rx, ry
        cnp.float_t dst, lev, dirc, Ia, Jx, Jy, norm, weight
        cnp.float_t[:] gradU
        cnp.float_t[:] gradI

    cdef cnp.uint16_t h = image.shape[0], w = image.shape[1]
    Ia, Jx, Jy, norm = 0, 0, 0, 0
    image_asfloat = image.astype(np.float, order='C')

    gradU = grad_func(i, j, flag_view, u_view, channel=1)

    indices = np.transpose(np.where(disk(epsilon)))
    center_ind_view = (indices - [epsilon, epsilon] + [i, j]).astype(np.int8, order='C')

    for x in range(center_ind_view.shape[0]):
        i_nb = center_ind_view[x, 0]
        j_nb = center_ind_view[x, 1]

        if i_nb <= 1 or i_nb >= h - 1 or j_nb <= 1 or j_nb >= w - 1:
            continue
        if flag_view[i_nb, j_nb] != KNOWN:
            continue

        rx = i - i_nb
        ry = j - j_nb

        dst = 1. / ((rx * rx + ry * ry) *
                    sqrt((rx * rx + ry * ry)))
        lev = 1. / (1 + abs(u_view[i_nb, j_nb] - u_view[i, j]))
        dirc = abs(rx * gradU[0] + ry * gradU[1])

        if dirc <= 0.01:
            dirc = 1.0e-6
        weight = dst * lev * dirc

        gradI = grad_func(i_nb, j_nb, flag_view, image_asfloat,
                                    channel=0)

        Ia += weight * image_view[i_nb, j_nb]
        Jx -= weight * gradI[0] * rx
        Jy -= weight * gradI[1] * ry
        norm += weight

    sat = (Ia / norm + (Jx + Jy) / (sqrt(Jx * Jx + Jy * Jy) + 1.0e-20) + 0.5)

    image_view[i, j] = int(round(sat))
