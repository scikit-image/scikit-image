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
                s = (u11 + u22 + r) * .5
                if s >= u11 and s >= u22:
                    sol = s
        else:
            sol = 1 + u11
    elif flag[i2, j2] == KNOWN:
        sol = 1 + u22

    return sol


def fast_marching_method(image, flag, u, heap, negate, epsilon=5):
    """Fast Marching Method implementation based on the algorithm outlined in
    the paper by Telea.
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
    """
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

    flag = _heap.init_flag(mask)

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
