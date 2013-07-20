import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt
from skimage.morphology import disk


__all__ = ['grad_func', 'ep_neighbor', 'inpaint_point']


KNOWN = 0
BAND = 1
INSIDE = 2


cdef cnp.float_t[:] grad_func(Py_ssize_t i, Py_ssize_t j,
                              cnp.uint8_t[:, ::1] flag_view,
                              cnp.float_t[:, ::1] array_view,
                              Py_ssize_t channel=1):
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

    cdef:
        cnp.float_t factor,
        cnp.float_t[:] gradU = np.zeros(2, dtype=np.float)

    if channel == 0:
        factor = 2.0
    elif channel == 1:
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


cdef ep_neighbor(Py_ssize_t i, Py_ssize_t j, Py_ssize_t h, Py_ssize_t w,
                Py_ssize_t epsilon):
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
        cnp.int8_t[:, ::1] center_ind
        Py_ssize_t i_ep, j_ep
        list nb = list()

    indices = np.transpose(np.where(disk(epsilon)))
    center_ind = np.ascontiguousarray(indices - [epsilon, epsilon] + [i, j], dtype=np.int8)

    for i_ep, j_ep in center_ind:
        if i_ep > 0 and j_ep > 0:
            if i_ep < h - 1 and j_ep < w - 1:
                nb.append([i_ep, j_ep])

    return nb


cpdef cnp.uint8_t[:, ::1] inpaint_point(Py_ssize_t i, Py_ssize_t j,
                                    cnp.uint8_t[:, ::1] image_view,
                                    cnp.uint8_t[:, ::1] flag_view,
                                    cnp.float_t[:, ::1] u_view,
                                    Py_ssize_t epsilon):
    """This function performs the actual inpainting operation. Inpainting
    involves "filling in" color in regions with unkown intensity values using
    the intensity and gradient information of surrounding known region.

    For further implementation details, see Section 2.3 and Figure 5 in [1]_

    Parameters
    ---------
    i, j : int
        Row and column index value of the pixel to be Inpainted
    image_view : array
        Padded single channel input image
    flag_view : array
        Array marking pixels as known, along the boundary to be solved, or
        inside the unknown region: 0 = KNOWN, 1 = BAND, 2 = INSIDE
    u_view : array
            The distance/time map from the boundary to each pixel.
    epsilon : integer
            Neighbourhood of (i, j) to be considered for inpainting

    Returns
    ------
    image_view[i, j] : integer
        Inpainted intensity value

    References
    ----------
    .. [1] Telea, A., "An Image Inpainting Technique based on the Fast Marching
            Method", Journal of Graphic Tools (2004).
            http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    """

    cdef:
        Py_ssize_t i_nb, j_nb
        cnp.int8_t rx, ry
        cnp.float_t dst, lev, dirc
        cnp.float_t Ia = 0, weight, Jx = 0, Jy = 0, norm = 0, sat
        cnp.float_t[:] gradU = np.zeros(2, dtype=np.float)
        cnp.float_t[:] gradI = np.zeros(2, dtype=np.float)

    gradU = grad_func(i, j, flag_view, u_view, channel=1)
    nb = ep_neighbor(i, j, image_view.shape[0], image_view.shape[1], epsilon)

    for i_nb, j_nb in nb:
        if flag_view[i_nb, j_nb] == KNOWN:
            rx = i - i_nb
            ry = j - j_nb

            dst = 1. / ((rx * rx + ry * ry) *
                        sqrt((rx * rx + ry * ry)))
            lev = 1. / (1 + abs(u_view[i_nb, j_nb] - u_view[i, j]))
            dirc = rx * gradU[0] + ry * gradU[1]

            if abs(dirc) <= 0.01:
                dirc = 1.0e-6
            weight = abs(dst * lev * dirc)

            gradI = grad_func(i_nb, j_nb, flag_view, np.ascontiguousarray(image_view, np.float), channel=0)

            Ia += weight * image_view[i_nb, j_nb]
            Jx -= weight * gradI[0] * rx
            Jy -= weight * gradI[1] * ry
            norm += weight

    sat = (Ia / norm + (Jx + Jy) / (sqrt(Jx * Jx + Jy * Jy) + 1.0e-20)
            + 0.5)
    image_view[i, j] = <cnp.uint8_t> round(sat)

    return image_view
