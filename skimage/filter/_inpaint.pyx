#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
from libc.math cimport sqrt

import numpy as np
from heapq import heappop, heappush
from skimage.morphology import disk


cdef:
    cnp.uint8_t KNOWN = 0
    cnp.uint8_t BAND = 1
    cnp.uint8_t INSIDE = 2


cdef cnp.float_t[:] grad_func(Py_ssize_t i, Py_ssize_t j,
                              cnp.uint8_t[:, ::1] flag_view,
                              cnp.float_t[:, ::1] array_view,
                              Py_ssize_t channel=1):
    """This function calculates the gradient of the distance/image of a pixel
    depending on the value of the flag of its neighbours. The gradient
    is computed using Central Differences method.

    This function is used to compute the gradient of intensity value, I and
    also the ``u`` value.

    Parameters
    ---------
    i, j : int
        Row and column index value of the pixel whose gradient is to be
        calculated
    flag_view : memory view, cnp.uint8_t
        Array marking pixels as known, along the boundary to be solved, or
        inside the unknown region: 0 = KNOWN, 1 = BAND, 2 = INSIDE
    array_view : memory view
        Either ``image_view`` or ``u_view``
    channel : integer
        If channel == 1 then the gradient of ``u`` is calculated
        If channel == 0 then the gradient of ``image`` is calculated

    Returns
    -------
    gradU : array, float
        The signed gradient of `image` or `u` depending on ``channel``.
        shape = (1, 2)

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


cdef inpaint_point(Py_ssize_t i, Py_ssize_t j, cnp.uint8_t[:, ::1] image_view,
                   cnp.uint8_t[:, ::1] flag_view, cnp.float_t[:, ::1] u_view,
                   Py_ssize_t neighbor):
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
    neighbor : integer
        Neighbourhood of (i, j) to be considered for inpainting

    Returns
    -------
    image[i, j] : integer
        Inpainted intensity value

    References
    ---------
    .. [1] Telea, A., "An Image Inpainting Technique based on the Fast Marching
            Method", Journal of Graphic Tools (2004).
            http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    """

    cdef:
        cnp.uint8_t[:, ::1] nb_view
        cnp.int16_t[:, ::1] center_ind_view
        cnp.int16_t i_nb, j_nb
        cnp.int8_t rx, ry
        cnp.float_t dst, lev, dirc, Ia, Jx, Jy, norm, weight
        cnp.float_t[:] gradU
        cnp.float_t[:] gradI

    cdef cnp.uint16_t h = image_view.shape[0], w = image_view.shape[1]
    Ia, Jx, Jy, norm = 0, 0, 0, 0
    image_asfloat = np.asarray(image_view, dtype=np.float)

    gradU = grad_func(i, j, flag_view, u_view, channel=1)

    indices = np.transpose(np.where(disk(neighbor)))
    center_ind_view = (indices - [neighbor, neighbor] + [i, j]).astype(np.int16,
                                                                     order='C')

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

        gradI = grad_func(i_nb, j_nb, flag_view, image_asfloat, channel=0)

        Ia += weight * image_view[i_nb, j_nb]
        Jx -= weight * gradI[0] * rx
        Jy -= weight * gradI[1] * ry
        norm += weight

    sat = (Ia / norm + (Jx + Jy) / (sqrt(Jx * Jx + Jy * Jy) + 1.0e-20) + 0.5)

    image_view[i, j] = int(round(sat))


cdef cnp.float_t eikonal(Py_ssize_t i1, Py_ssize_t j1, Py_ssize_t i2,
                         Py_ssize_t j2, cnp.uint8_t[:, ::1] flag_view,
                         cnp.float_t[:, ::1] u_view):
    """Solve a step of the Eikonal equation.

    The `u` values of known pixels (marked by `flag`) are considered for
    computing the `u` value of the neighbouring pixel.

    See Equation 4 and Figure 4 in [1]_ for implementation details.

    Parameters
    ----------
    i1, j1, i2, j2 : int
        Row and column indices of two diagonally-adjacent pixels.
    flag : array
        Array marking pixels as known, along the boundary to be solved, or
        inside the unknown region: 0 = KNOWN, 1 = BAND, 2 = INSIDE
    u : array
        The distance/time map from the boundary to each pixel.

    Returns
    -------
    u_out : float
        The `u` value for the pixel `(i2, j1)`.

    Notes
    -----
    The boundary is assumed to move with a constant speed in a direction normal
    to the boundary at all pixels, such that the time of arrival `u` must be
    monotonically increasing. Note that `u` is often denoted `T`.

    References
    ----------
    .. [1] Telea, A., "An Image Inpainting Technique based on the Fast Marching
            Method", Journal of Graphic Tools (2004).
            http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    """

    cdef cnp.float_t u_out, u1, u2, r, s

    u_out = 1.0e6
    u1 = u_view[i1, j1]
    u2 = u_view[i2, j2]

    if flag_view[i1, j1] == KNOWN:
        if flag_view[i2, j2] == KNOWN:
            r = sqrt(2 - (u1 - u2) ** 2)
            s = (u1 + u2 - r) * 0.5
            if s >= u1 and s >= u2:
                u_out = s
            else:
                s += r
                if s >= u1 and s >= u2:
                    u_out = s
        else:
            u_out = 1 + u1
    elif flag_view[i2, j2] == KNOWN:
        u_out = 1 + u2

    return u_out


cpdef fast_marching_method(cnp.uint8_t[:, ::1] image_view,
                           cnp.uint8_t[:, ::1] flag_view,
                           cnp.float_t[:, ::1] u_view, heap,
                           _run_inpaint=True, Py_ssize_t neighbor=5):
    """Inpaint an image using the Fast Marching Method (FMM).

    Image Inpainting technique based on the Fast Marching Method implementation
    as described in [1]_. FMM is used for computing the evolution of
    boundary moving in a direction *normal* to itself.

    Parameters
    ---------
    image : array
        Input image padded by a single row/column on all sides
    flag : array
        Array marking pixels as known, along the boundary to be solved, or
        inside the unknown region: 0 = KNOWN, 1 = BAND, 2 = INSIDE
    u : array
        The distance/time map from the boundary to each pixel.
    heap : list of tuples
        Priority heap which stores pixels for processing.
    _run_inpaint : bool
        If `True` then inpaint the image
        If `False` then only compute the distance/time map, ``u``
    neighbor : integer
        Neighbourhood of the pixel of interest

    Returns
    ------
    image or u : array
        The inpainted image or distance/time map depending on `_run_inpaint`.

    Notes
    -----
    The steps of the algorithm are as follows:
    - Extract the pixel with the smallest `u` value in the BAND pixels
    - Update its `flag` value as KNOWN
    - March the boundary inwards by adding new points.
        - If they are either INSIDE or BAND, compute its `u` value using the
          `eikonal` function for all the 4 quadrants
        - If `flag` is INSIDE
            - Change it to BAND
            - Inpaint the pixel
        - Select the `min` value and assign it as the `u` value of the pixel
        - Insert this new value in the `heap`

    For further details, see [1]_

    References
    ----------
    .. [1] Telea, A., "An Image Inpainting Technique based on the Fast Marching
            Method", Journal of Graphic Tools (2004).
            http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    """

    cdef Py_ssize_t i, j, i_nb, j_nb

    while len(heap):
        i, j = heappop(heap)[1]
        flag_view[i, j] = KNOWN

        if ((i <= 1) or (j <= 1) or (i >= image_view.shape[0] - 2)
                or (j >= image_view.shape[1] - 2)):
            continue

        for (i_nb, j_nb) in (i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1):

            if not flag_view[i_nb, j_nb] == KNOWN:
                u_view[i_nb, j_nb] = min(eikonal(i_nb - 1, j_nb, i_nb,
                                                 j_nb - 1, flag_view, u_view),
                                         eikonal(i_nb + 1, j_nb, i_nb,
                                                 j_nb - 1, flag_view, u_view),
                                         eikonal(i_nb - 1, j_nb, i_nb,
                                                 j_nb + 1, flag_view, u_view),
                                         eikonal(i_nb + 1, j_nb, i_nb,
                                                 j_nb + 1, flag_view, u_view))

                if flag_view[i_nb, j_nb] == INSIDE:
                    flag_view[i_nb, j_nb] = BAND
                    heappush(heap, (u_view[i_nb, j_nb], (i_nb, j_nb)))

                    if _run_inpaint:
                        inpaint_point(i_nb, j_nb, image_view, flag_view,
                                      u_view, neighbor)
