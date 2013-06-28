__all__ = ['fast_marching_method', 'eikonal_solve']

import numpy as np
import _heap
import _inpaint
from heapq import heappop, heappush

BAND = 0
KNOWN = 1
INSIDE = 2


def fast_marching_method(image, mask, epsilon):
    """Fast Marching Method implementation based on the algorithm outlined in
    the paper by Telea.
    """

    heap = []

    flag, u = _heap.generate_flags(mask)
    heap = _heap.generate_heap(flag, u)

    while len(heap):
        item = heappop(heap)
        i, j = item.index

        if ((i <= 1) or (j <= 1) or (i > mask.shape[0] - 1)
                or (j > mask.shape[1] - 1)):
            continue

        for (k, l) in (i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1):
            if flag[k, l] is not KNOWN and flag[k, l] is INSIDE:
                _inpaint.inpaint_point(k, l, image, flag, u, epsilon)
                flag[k, l] = BAND
                #below indent-1
                u[k, l] = min(eikonal_solve(k - 1, l, k, l - 1, flag, u),
                              eikonal_solve(k + 1, l, k, l - 1, flag, u),
                              eikonal_solve(k - 1, l, k, l + 1, flag, u),
                              eikonal_solve(k + 1, l, k, l + 1, flag, u))
                heappush(heap, [u[k, l], (k, l)])

        flag[i, j] = KNOWN
        u[i, j] = -u[i, j]


def eikonal_solve(i1, j1, i2, j2, flag, u):
    """This function provides the solution for the Eikonal equation.
    """

    sol = 1.0e6
    t11 = u[i1, j1]
    t22 = u[i2, j2]
    mint = min(t11, t22)
    if flag[i1, j1] is not INSIDE:
        if flag[i2, j2] is not INSIDE:
            if abs(t11 - t22) >= 1.0:
                sol = 1 + mint
            else:
                sol = (t11 + t22 + np.sqrt(2 - (t11 - t22) * (t11 - t22))) * .5
        else:
            sol = 1 + t11
    elif flag[i2, j2] is not INSIDE:
        sol = 1 + t22
    else:
        sol = 1 + mint

    return sol
