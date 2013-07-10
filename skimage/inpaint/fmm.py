import numpy as np
from _inpaint import inpaint_point as inp_point
from heapq import heappop, heappush
import _heap


__all__ = ['inpaint', 'fast_marching_method', 'eikonal']


KNOWN = 0
BAND = 1
INSIDE = 2


def eikonal(i1, j1, i2, j2, flag, u):
    """Solve a step of the Eikonal equation.

    The `u` values of known pixels (marked by `flag`) are considered for
    computing the `u` value of the neighbouring pixel.

    See Equation 4 and Figure 4 in [1]_ for implementation details.

    Parameters
    ---------
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

    u_out = 1.0e6
    u1 = u[i1, j1]
    u2 = u[i2, j2]

    if flag[i1, j1] == KNOWN:
        if flag[i2, j2] == KNOWN:
            r = np.sqrt(2 - (u1 - u2) ** 2)
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
        u_out = 1 + u2  # Instead of u2 [1]_ uses u[i1, j2]. Typo in paper?
    return u_out


def fast_marching_method(image, flag, u, heap, _run_inpaint=True, epsilon=5):
    """Inpaint an image using the Fast Marching Method (FMM).

    Image Inpainting technique based on the Fast Marching Method implementation
    as described in [1]_. FMM is used for computing the evolution of
    boundary moving in a direction *normal* to itself.

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
        If `True` then used for initialising `u` values outside the BAND
        If `False` then used to inpaint the pixels marked as INSIDE
    epsilon : integer
        Neighbourhood of the pixel of interest

    Returns
    ------
    image or u : array
        The inpainted image or distance/time map depending on `_run_inpaint`.

    References
    ----------
    .. [1] Telea, A., "An Image Inpainting Technique based on the Fast Marching
           Method", Journal of Graphic Tools (2004).
           http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    """

    while len(heap):
        i, j = heappop(heap)[1]
        flag[i, j] = KNOWN

        if ((i <= 1) or (j <= 1) or (i >= image.shape[0] - 2)
                or (j >= image.shape[1] - 2)):
            continue

        for (i_nb, j_nb) in (i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1):
            if not flag[i_nb, j_nb] == KNOWN:
                u[i_nb, j_nb] = min(eikonal(i_nb - 1, j_nb,
                                            i_nb, j_nb - 1, flag, u),
                                    eikonal(i_nb + 1, j_nb,
                                            i_nb, j_nb - 1, flag, u),
                                    eikonal(i_nb - 1, j_nb,
                                            i_nb, j_nb + 1, flag, u),
                                    eikonal(i_nb + 1, j_nb,
                                            i_nb, j_nb + 1, flag, u))

                if negate is False:
                    image[i_nb - 1, j_nb - 1] = inp_point(i_nb, j_nb, image,
                                                          flag, u, epsilon)

                if flag[i_nb, j_nb] == INSIDE:
                    flag[i_nb, j_nb] = BAND
                    heappush(heap, (u[i_nb, j_nb], (i_nb, j_nb)))

                    if _run_inpaint:
                        inp_point(i_nb, j_nb, image, flag, u, epsilon)

                # heappush(heap, (u[i_nb, j_nb], (i_nb, j_nb)))

        if not _run_inpaint:
            u[i, j] = -u[i, j]

    if not _run_inpaint:
        return u
    else:
        return image


def inpaint(input_image, inpaint_mask, epsilon=5):
    """Inpaint image in areas specified by a mask.

    Parameters
    ---------
    input_image : array
        This can be either a single channel or three channel image.
    inpaint_mask : array, bool
        Mask containing pixels to be inpainted. `True` values are inpainted.
    epsilon : int
        Determining the range of the neighbourhood for inpainting a pixel

    Returns
    ------
    painted : array
        The inpainted image.

    References
    ---------
    .. [1] Telea, A., "An Image Inpainting Technique based on the Fast Marching
           Method", Journal of Graphic Tools (2004).
           http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    """
    # TODO: Error checks. Image either 3 or 1 channel. All dims same

    h, w = input_image.shape
    image = np.zeros((h + 2, w + 2), np.uint8)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    image[1: -1, 1: -1] = input_image
    mask[1: -1, 1: -1] = inpaint_mask

    flag, u, heap = _heap.initialise(mask)

    painted = fast_marching_method(image, flag, u, heap, epsilon=epsilon)

    return painted[1:-1, 1:-1]
