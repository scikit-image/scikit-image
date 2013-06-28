__all__ = ['fast_marching_method', 'eikonal']

import numpy as np
import _heap
import _inpaint
from heapq import heappop, heappush

KNOWN = 0
BAND = 1
INSIDE = 2


def fast_marching_method(image, mask, epsilon, negate):
    """Fast Marching Method implementation based on the algorithm outlined in
    the paper by Telea.
    """

    heap = []

    flag, u = _heap.generate_flags(mask)
    heap = _heap.generate_heap(flag, u)

    while len(heap):
        i, j = heappop(heap)[1]
        flag[i, j] = KNOWN

        if ((i <= 1) or (j <= 1) or (i > mask.shape[0] - 1)
                or (j > mask.shape[1] - 1)):
            continue

        for (i_nb, j_nb) in (i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1):
            if flag[i_nb, j_nb] is INSIDE:
                u[i_nb, j_nb] = min(eikonal(i_nb - 1, j_nb,
                                            i_nb, j_nb - 1, flag, u),
                                    eikonal(i_nb + 1, j_nb,
                                            i_nb, j_nb - 1, flag, u),
                                    eikonal(i_nb - 1, j_nb,
                                            i_nb, j_nb + 1, flag, u),
                                    eikonal(i_nb + 1, j_nb,
                                            i_nb, j_nb + 1, flag, u))
                if negate is False:
                    _inpaint.inpaint_point(i_nb, j_nb, image, flag, u, epsilon)

                flag[i_nb, j_nb] = BAND
                heappush(heap, [u[i_nb, j_nb], (i_nb, j_nb)])

        if negate is True:
            u[i, j] = -u[i, j]


def eikonal(i1, j1, i2, j2, flag, u):
    """This function provides the solution for the Eikonal equation.
    """

    sol = 1.0e6
    u11 = u[i1, j1]
    u22 = u[i2, j2]
    min_u = min(u11, u22)
    if flag[i1, j1] is not INSIDE and flag[i2, j2] is not INSIDE:
        if abs(u11 - u22) >= 1.0:
            sol = 1 + min_u
        else:
            sol = (u11 + u22 + np.sqrt(2 - (u11 - u22) * (u11 - u22))) * .5
    elif flag[i1, j1] is not INSIDE and flag[i2, j2] is INSIDE:
        sol = 1 + u11
    elif flag[i1, j1] is INSIDE and flag[i2, j2] is not INSIDE:
        sol = 1 + u22
    else:
        sol = 1 + min_u

    return sol
