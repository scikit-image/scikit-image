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


cdef cnp.float_t grad_func(Py_ssize_t i, Py_ssize_t j,
                           cnp.uint8_t[:, :] flag,
                           cnp.float_t[:, :] array,
                           cnp.float_t factor=0.5):
    """Return the x-gradient of the input array at a pixel.

    This gradient is structured to ignore inner, unknown regions as specified
    by the `flag` array. By default, this returns the gradient in the
    x-direction. To get the y-gradient, switch the order of `i`, `j`, and
    transpose `flag` and `array`.

    Parameters
    ---------
    i, j : int
        Row and column index of the pixel whose gradient is to be calculated.
    flag : array, uint8
        Array marking pixels as known, along the boundary to be solved, or
        inside the unknown region: 0 = KNOWN, 1 = BAND, 2 = INSIDE
    array : array, float
        Either ``image`` or ``u``, whose gradient is to be computed
    factor : float
        0.5 for ``array = u``, 2.0 for ``array = image``

    Returns
    -------
    grad : float
        The local gradient of array.

    """
    
    cdef:
        cnp.float_t grad

    if flag[i, j + 1] != INSIDE:
        if flag[i, j - 1] != INSIDE:
            grad = (array[i, j + 1] - array[i, j - 1]) * factor
        else:
            grad = (array[i, j + 1] - array[i, j])
    else:
        if flag[i, j - 1] != INSIDE:
            grad = (array[i, j] - array[i, j - 1])
        else:
            grad = 0

    return grad


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
        Already padded single channel input image
    flag : array, uint8
        Array marking pixels as known, along the boundary to be solved, or
        inside the unknown region: 0 = KNOWN, 1 = BAND, 2 = INSIDE
    u : array, float
        The distance/time map from the boundary to each pixel.
    radius : int
        Neighbourhood of (i, j) to be considered for inpainting

    Returns
    -------
    image[i, j] : float
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
        cnp.float_t gradx_u, grady_u, gradx_img, grady_img
        Py_ssize_t k
    cdef cnp.uint16_t h = image.shape[0], w = image.shape[1]

    Ia, Jx, Jy, norm = 0, 0, 0, 0

    gradx_u = grad_func(i, j, flag, u, factor=0.5)
    grady_u = grad_func(j, i, flag.T, u.T, factor=0.5)

    for k in range(shifted_indices.shape[0]):
        i_nb = shifted_indices[k, 0]
        j_nb = shifted_indices[k, 1]

        if i_nb <= 1 or i_nb >= h - 1 or j_nb <= 1 or j_nb >= w - 1:
            continue
        if flag[i_nb, j_nb] != KNOWN:
            continue

        ry = i - i_nb
        rx = j - j_nb

        # geometric_dst : more weightage to geometrically closer pixels
        # levelset_dst : more weightage to points with nearly same time map, ``u``
        # direction : dot product, displacement vector and gradient vector
        
        geometric_dst = 1. / ((rx * rx + ry * ry) * sqrt((rx * rx + ry * ry)))
        levelset_dst = 1. / (1 + abs(u[i_nb, j_nb] - u[i, j]))
        direction = abs(rx * gradx_u + ry * grady_u)

        # Small values of ``direction``, implies displacement vector and 
        # gradient vector nearly perpendicular, hence force low contribution
        if direction <= 0.01:
            direction = 1.0e-6
        weight = geometric_dst * levelset_dst * direction

        gradx_img = grad_func(i_nb, j_nb, flag, image, factor=2.0)
        grady_img = grad_func(j_nb, i_nb, flag.T, image.T, factor=2.0)

        Ia += weight * image[i_nb, j_nb]
        Jx -= weight * gradx_img * rx
        Jy -= weight * grady_img * ry
        norm += weight

    # Inpainted value considering the effect of gradient of intensity value
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
    u : array, float
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
    image : array, float
        Initial input image padded by a single row/column on all sides
    flag : array, uint8
        Array marking pixels as known, along the boundary to be solved, or
        inside the unknown region: 0 = KNOWN, 1 = BAND, 2 = INSIDE
    u : array, float
        The distance/time map from the boundary to each pixel.
    heap : list of tuples
        Priority heap which stores pixels for processing.
    _run_inpaint : bool
        If``True`` then inpaint the image
        If``False`` then only compute the distance/time map,``u``
    radius : int
        Neighbourhood of the pixel of interest

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
    indices_centered = np.ascontiguousarray((indices - radius), np.int16)

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
                                    eikonal(i_nb + 1, j_nb, i_nb, 
                                            j_nb - 1, flag, u),
                                    eikonal(i_nb - 1, j_nb, i_nb, 
                                            j_nb + 1, flag, u),
                                    eikonal(i_nb + 1, j_nb, i_nb, 
                                            j_nb + 1, flag, u))

                if flag[i_nb, j_nb] == INSIDE:
                    flag[i_nb, j_nb] = BAND
                    heappush(heap, (u[i_nb, j_nb], (i_nb, j_nb)))

                    if _run_inpaint:
                        shifted_indices[:, 0] = indices_centered[:, 0] + i_nb
                        shifted_indices[:, 1] = indices_centered[:, 1] + j_nb
                        
                        inpaint_point(i_nb, j_nb, image, flag,
                                      u, shifted_indices, radius)
