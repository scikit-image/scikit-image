__all__ = ['inpaint', 'fast_marching_method', 'eikonal']

import numpy as np
import _heap
from _inpaint import inpaint_point as inp_point
from heapq import heappop, heappush
from skimage.morphology import dilation, square

KNOWN = 0
BAND = 1
INSIDE = 2


def eikonal(i1, j1, i2, j2, flag, u):
    """This function provides the solution for the Eikonal equation.
    """

    sol = 1.0e6
    u11 = u[i1, j1]
    u22 = u[i2, j2]

    if flag[i1, j1] == KNOWN:
        if flag[i2, j2] == KNOWN:
            r = np.sqrt(2 - (u11 - u22) * (u11 - u22))
            s = (u11 + u22 - r) * .5
            if s >= u11 and s >= u22:
                sol = s
            else:
                s += r
                if s >= u11 and s >= u22:
                    sol = s
        else:
            sol = 1 + u11
    elif flag[i2, j2] == KNOWN:
        sol = 1 + u22

    return sol


def fast_marching_method(image, flag, u, heap, negate, epsilon=5):
    """Fast Marching Method implementation based on the algorithm outlined in
    Telea, A. (2004). *An Image Inpainting Technique based on the Fast Marching
    Method*. Journal of graphics tools, 9(1):23–34.

    Image Inpainting technique based on the Fast Marching Method implementation
    as described in the paper above. FMM is used for computing the evolution of
    boundary moving in a direction *normal* to itself. The tangential component
    of the velocity is not of interest here. This algorithm has been adapted by
    Telea for the purpose of inpainting. For this we assume that the speed is
    uniform at all the pixels and set it to 1. There are two phases to this
    implementation: the initialisation phase which is described in the
    `initialise` function and the propagation phase, which is described here.

    In order to obtain the correct representation it is important to propagate
    the boundary pixels to the interior (INSIDE) region in the increasing
    distance from the boundary. The solution to Eikonal equation helps
    achieve this.
    `|gradT| = 1`

    The steps of the algorithm are as follows:
    * Extract the pixel with the smallest `T` value in the BAND pixels. We use
            a priority heap implementation whose head is always the smallest
            value in the heap
        + `T[i, j] = heappop[0]`
    * Update its `flag` value as KNOWN
    * March the boundary inwards by adding new points.
            `for (k, l) in (i-1, j),(i+1, j), (i, j-1), (i, j+1)`
        + If they are either INSIDE or BAND, compute its `T` value using the
                `eikonal` function in all the 4 quadrants
        + If `flag` is INSIDE
            - Change it to BAND
            - Inpaint `(k, l)`
        + Select the `min` value and assign it as the `T` value of `(k, l)`
        + Insert this new value in the `heap`

    You can further check out the numerical approximation to the Eikonal
    equation and the algorithm in the paper.

    Parameters
    ---------
    image: ndarray of unsigned integers
        Input image padded by a single row/column on all sides
    flag: ndarray of unsigned integers
        Values are either 0 (KNOWN), 1(BAND) or 2(INSIDE)
    u: ndarray of float
        Consists of the `T` values of the pixels. 1.0e6 for pixel marked INSIDE
        else 0
    heap: list of tuples
        Priority Heap implementation based on the in-built Python`heapq` module
    negate: bool
        If `True` then used for initialising `u` values outside the BAND
        If `False` then used to inpaint the pixels marked as INSIDE
    epsilon: unsgned integer
        Neighbourhood of the pixel of interest

    Returns
    ------
    image: ndarray of unsigned integers
        Consists of the result after inpainting the all the pixels.

    References
    ----------
    .. [1] http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    """

    while len(heap):
        i, j = heappop(heap)[1]
        flag[i, j] = KNOWN

        if ((i <= 1) or (j <= 1) or (i >= image.shape[0] - 1)
                or (j >= image.shape[1] - 1)):
            continue

        for (i_nb, j_nb) in (i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1):
            if flag[i_nb, j_nb] != KNOWN:

                u[i_nb, j_nb] = min(eikonal(i_nb - 1, j_nb,
                                            i_nb, j_nb - 1, flag, u),
                                    eikonal(i_nb + 1, j_nb,
                                            i_nb, j_nb - 1, flag, u),
                                    eikonal(i_nb - 1, j_nb,
                                            i_nb, j_nb + 1, flag, u),
                                    eikonal(i_nb + 1, j_nb,
                                            i_nb, j_nb + 1, flag, u))

                if flag[i_nb, j_nb] == INSIDE:
                    flag[i_nb, j_nb] = BAND
                    if negate is False:
                        inp_point(i_nb, j_nb, image, flag, u, epsilon)
                    #   inp_point(i, j, image, flag, u, epsilon)

                heappush(heap, [u[i_nb, j_nb], (i_nb, j_nb)])

        if negate is True:
            u[i, j] = -u[i, j]

    if negate is True:
        return u
    else:
        return image


def inpaint(input_image, inpaint_mask, epsilon=5):
    """Wrapper function for the inpainitng techique based on Fast Marching
    Method as introduced by Telea. Check out the `fast_marching_method`
    function for details regarding the Fast Marching Method implementation
    and `inpaint_point` for details regarding the Inpainting algorithm.

    Parameters
    ---------
    input_image: ndarray of unsigned integers
        This can be either a single channel or three channel image.
    inpaint_mask: ndarray of bool
        Mask containing the pixels to be inpainted. `True` values are to
        be inpainted
    epsilon: unsigned integer
        Determining the range of the neighbourhood for inpainting a pixel

    Returns
    ------
    output: ndarray of unsigned integers
        Contains the final inpainted output image

    References
    ---------
    . [1] http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    """
    # TODO: Error checks. Image either 3 or 1 channel. All dims same
    # if input_image.ndim == 3:
    #     m, n, channel = input_image.shape
    #     image = np.zeros((m + 2, n + 2, channel), np.uint8)
    # else:
    #     m, n = input_image.shape
    #     image = np.zeros((m + 2, n + 2), np.uint8)
    m, n = input_image.shape
    image = np.zeros((m + 2, n + 2), np.uint8)
    mask = np.zeros((m + 2, n + 2), bool)
    image[1: -1, 1: -1] = input_image
    mask[1: -1, 1: -1] = inpaint_mask

    flag = np.zeros_like(image, dtype=np.uint8)
    u = np.zeros_like(image, dtype=float)
    heap = []

    outside = dilation(mask, square(2 * epsilon + 1))
    outside_band = np.logical_xor(outside, mask).astype(np.uint8)
    out_flag = _heap.init_flag(mask)
    u = _heap.init_u(flag)

    out_heap = []
    _heap.generate_heap(out_heap, out_flag, u)
    u = fast_marching_method(outside_band, out_flag, u, out_heap, negate=True)

    heap = []
    _heap.generate_heap(heap, flag, u)
    output = fast_marching_method(image, flag, u, heap, negate=False,
                                  epsilon=epsilon)

    return output
