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
                              cnp.uint8_t[:, ::1] flag,
                              cnp.float_t[:, ::1] array,
                              cnp.float_t factor=0.5):
    """This function calculates the gradient of the distance/image of a pixel
    depending on the value of the flag of its neighbours. The gradient
    is computed using Central Differences method.

    This function is used to compute the gradient of intensity value, ``image``
    and also the ``u`` value.

    Parameters
    ---------
    i, j : int
        Row and column index value of the pixel whose gradient is to be
        calculated
    flag : array, cnp.uint8_t
        Array marking pixels as known, along the boundary to be solved, or
        inside the unknown region: 0 = KNOWN, 1 = BAND, 2 = INSIDE
    array : array
        Either ``image`` or ``u``
    factor : float
        ``factor`` = 0.5 for the gradient of ``u``
        ``factor`` = 2.0 for the gradient of ``image``

    Returns
    -------
    gradU : array, float
        The signed gradient of ``image`` or ``u`` depending on ``factor``.
        1D array with 2 elements. First element represents gradient in X
        direction and second in the Y direction.

    """

    cdef:
        cnp.float_t[:] gradU = np.zeros(2, dtype=np.float)

    if flag[i, j + 1] != INSIDE:
        if flag[i, j - 1] != INSIDE:
            gradU[0] = (array[i, j + 1] - array[i, j - 1]) * factor
        else:
            gradU[0] = (array[i, j + 1] - array[i, j])
    else:
        if flag[i, j - 1] != INSIDE:
            gradU[0] = (array[i, j] - array[i, j - 1])
        else:
            gradU[0] = 0

    if flag[i + 1, j] != INSIDE:
        if flag[i - 1, j] != INSIDE:
            gradU[1] = (array[i + 1, j] - array[i - 1, j]) * factor
        else:
            gradU[1] = (array[i + 1, j] - array[i, j])
    else:
        if flag[i - 1, j] != INSIDE:
            gradU[1] = (array[i, j] - array[i - 1, j])
        else:
            gradU[1] = 0

    return gradU


cdef inpaint_point(cnp.int16_t i, cnp.int16_t j, cnp.float_t[:, ::1] image,
                   cnp.uint8_t[:, ::1] flag, cnp.float_t[:, ::1] u,
                   cnp.int16_t[:, ::1] shifted_indices, Py_ssize_t radius):
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
    radius : integer
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
        cnp.uint8_t[:, ::1] nb
        cnp.int16_t i_nb, j_nb
        cnp.int8_t rx, ry
        cnp.float_t geometric_dst, levelset_dst, direction
        cnp.float_t Ia, Jx, Jy, norm, weight
        cnp.float_t[:] gradU
        cnp.float_t[:] gradI
        Py_ssize_t k
    cdef cnp.uint16_t h = image.shape[0], w = image.shape[1]

    Ia, Jx, Jy, norm = 0, 0, 0, 0

    gradU = grad_func(i, j, flag, u, factor=0.5)

    for k in range(shifted_indices.shape[0]):
        i_nb = shifted_indices[k, 0]
        j_nb = shifted_indices[k, 1]

        if i_nb <= 1 or i_nb >= h - 1 or j_nb <= 1 or j_nb >= w - 1:
            continue
        if flag[i_nb, j_nb] != KNOWN:
            continue

        ry = i - i_nb
        rx = j - j_nb

        geometric_dst = 1. / ((rx * rx + ry * ry) * sqrt((rx * rx + ry * ry)))
        levelset_dst = 1. / (1 + abs(u[i_nb, j_nb] - u[i, j]))
        direction = abs(rx * gradU[0] + ry * gradU[1])

        if direction <= 0.01:
            direction = 1.0e-6
        weight = geometric_dst * levelset_dst * direction

        gradI = grad_func(i_nb, j_nb, flag, image, factor=2.0)

        Ia += weight * image[i_nb, j_nb]
        Jx -= weight * gradI[0] * rx
        Jy -= weight * gradI[1] * ry
        norm += weight

    image[i, j] = (Ia / norm + (Jx + Jy) / (sqrt(Jx * Jx + Jy * Jy) + 1.0e-20)
                   + 0.5)


cdef cnp.float_t eikonal(Py_ssize_t i1, Py_ssize_t j1, Py_ssize_t i2,
                         Py_ssize_t j2, cnp.uint8_t[:, ::1] flag,
                         cnp.float_t[:, ::1] u):
    """Solve a step of the Eikonal equation.

    The``u`` values of known pixels (marked by``flag``) are considered for
    computing the``u`` value of the neighbouring pixel.

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
        The``u`` value for the pixel``(i2, j1)``.

    Notes
    -----
    The boundary is assumed to move with a constant speed in a direction normal
    to the boundary at all pixels, such that the time of arrival``u`` must be
    monotonically increasing. Note that``u`` is often denoted``T``.

    References
    ----------
    .. [1] Telea, A., "An Image Inpainting Technique based on the Fast Marching
            Method", Journal of Graphic Tools (2004).
            http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    """

    cdef cnp.float_t u_out, u1, u2, r, s

    u_out = 1.0e6
    u1 = u[i1, j1]
    u2 = u[i2, j2]

    if flag[i1, j1] == KNOWN:
        if flag[i2, j2] == KNOWN:
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
    elif flag[i2, j2] == KNOWN:
        u_out = 1 + u2

    return u_out


cpdef fast_marching_method(cnp.float_t[:, ::1] image,
                           cnp.uint8_t[:, ::1] flag,
                           cnp.float_t[:, ::1] u, heap,
                           _run_inpaint=True, Py_ssize_t radius=5):
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
        If``True`` then inpaint the image
        If``False`` then only compute the distance/time map,``u``
    radius : integer
        Neighbourhood of the pixel of interest

    Returns
    ------
    image or u : array
        The inpainted image or distance/time map depending on ``_run_inpaint``.

    Notes
    -----
    The steps of the algorithm are as follows:
    - Extract the pixel with the smallest ``u`` value in the BAND pixels
    - Update its ``flag`` value as KNOWN
    - March the boundary inwards by adding new points.
        - If they are either INSIDE or BAND, compute its ``u`` value using the
          ``eikonal`` function for all the 4 quadrants
        - If ``flag`` is INSIDE
            - Change it to BAND
            - Inpaint the pixel
        - Select the ``min`` value and assign it as ``u`` value of the pixel
        - Insert this new value in the ``heap``

    For further details, see [1]_

    References
    ----------
    .. [1] Telea, A., "An Image Inpainting Technique based on the Fast Marching
            Method", Journal of Graphic Tools (2004).
            http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    """

    cdef:
        Py_ssize_t i, j,
        cnp.int16_t i_nb, j_nb
        cnp.int16_t[:, ::1] shifted_indices

    indices = np.transpose(np.where(disk(radius)))
    indices_centered = (indices - [radius, radius]).astype(np.int16, order='C')

    while len(heap):
        i, j = heappop(heap)[1]
        flag[i, j] = KNOWN

        if ((i <= 1) or (j <= 1) or (i >= image.shape[0] - 2)
                or (j >= image.shape[1] - 2)):
            continue

        for (i_nb, j_nb) in (i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1):

            if not flag[i_nb, j_nb] == KNOWN:
                u[i_nb, j_nb] = min(eikonal(i_nb - 1, j_nb, i_nb,
                                            j_nb - 1, flag, u),
                eikonal(i_nb + 1, j_nb, i_nb, j_nb - 1, flag, u),
                eikonal(i_nb - 1, j_nb, i_nb, j_nb + 1, flag, u),
                eikonal(i_nb + 1, j_nb, i_nb, j_nb + 1, flag, u))

                if flag[i_nb, j_nb] == INSIDE:
                    flag[i_nb, j_nb] = BAND
                    heappush(heap, (u[i_nb, j_nb], (i_nb, j_nb)))

                    if _run_inpaint:
                        shifted_indices = indices_centered
                        + np.asarray([i_nb, j_nb], np.int16)
                        inpaint_point(i_nb, j_nb, image, flag,
                                      u, shifted_indices, radius)
