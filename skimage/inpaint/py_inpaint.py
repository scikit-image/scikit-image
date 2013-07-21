import numpy as np
from skimage.morphology import disk


__all__ = ['grad_func', 'ep_neighbor', 'inpaint_point']


KNOWN = 0
BAND = 1
INSIDE = 2


def grad_func(i, j, flag, array, channel=1):
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
        If channel == 1 then the gradient of `u` is calculated
        If channel == 0 then the gradient of `image` is calculated

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
    elif channel == 1:
        u = np.array(array, float)
        factor = 0.5

    if flag_view[i, j + 1] != INSIDE and flag_view[i, j - 1] != INSIDE:
        gradU[0] = (array_view[i, j + 1] - array_view[i, j - 1]) * factor
    elif flag_view[i, j + 1] != INSIDE and flag_view[i, j - 1] == INSIDE:
        gradU[0] = (array_view[i, j + 1] - array_view[i, j])
    elif flag_view[i, j + 1] == INSIDE and flag_view[i, j - 1] != INSIDE:
        gradU[0] = (array_view[i, j] - array_view[i, j - 1])
    elif flag_view[i, j + 1] == INSIDE and flag_view[i, j - 1] == INSIDE:
        gradU[0] = 0

    if flag_view[i + 1, j] != INSIDE and flag_view[i - 1, j] != INSIDE:
        gradU[1] = (array_view[i + 1, j] - array_view[i - 1, j]) * factor
    elif flag_view[i + 1, j] != INSIDE and flag_view[i - 1, j] == INSIDE:
        gradU[1] = (array_view[i + 1, j] - array_view[i, j])
    elif flag_view[i + 1, j] == INSIDE and flag_view[i - 1, j] != INSIDE:
        gradU[1] = (array_view[i, j] - array_view[i - 1, j])
    elif flag_view[i + 1, j] == INSIDE and flag_view[i - 1, j] == INSIDE:
        gradU[1] = 0

    return gradU


cdef cnp.uint8_t[:, ::1] ep_neighbor(cnp.int8_t[:, ::1] center_ind_view, Py_ssize_t h, Py_ssize_t w):
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

    cdef:
        Py_ssize_t i_ep, j_ep

    nb = []

    for i_ep, j_ep in center_ind_view:
        if i_ep > 0 and j_ep > 0:
            if i_ep < h - 1 and j_ep < w - 1:
                nb.append([i_ep, j_ep])

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
    flag_view : array
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
        cnp.uint8_t i_nb, j_nb
        cnp.int8_t rx, ry
        cnp.float_t dst, lev, dirc, Ia, Jx, Jy, norm, weight
        cnp.float_t[:] gradU
        cnp.float_t[:] gradI

    Ia, Jx, Jy, norm = 0, 0, 0, 0
    gradU = grad_func(i, j, flag_view, u_view, channel=1)

    indices = np.transpose(np.where(disk(epsilon)))
    center_ind_view = (indices - [epsilon, epsilon] + [i, j]).astype(np.int8, order='C')

    nb_view = ep_neighbor(center_ind_view, image.shape[0], image.shape[1])

    for i_nb, j_nb in nb_view:
        if flag_view[i_nb, j_nb] == KNOWN:
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
