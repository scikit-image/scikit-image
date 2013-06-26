__all__ = ['fast_marching_method', 'eikonal_solve']

import numpy as np
import _heap
import _inpaint
from heapq import heappop, heappush


BAND = 0
KNOWN = 1
INSIDE = 2

def fast_marching_method(image, mask):
    """Fast Marching Method implementation based on the algorithm outlined in
    the paper by Telea.
    """

    heap = []

    flag, T = _heap.generate_flags(mask)
    heap = _heap.generate_heap(flag, T)

    while len(heap):
        item = heappop(heap)
        i, j = item.index

        if ((i <= 1) or (j <= 1) or (i > mask.shape[0]-1)
                or (j > mask.shape[1]-1)):
            continue
        for (k, l) in (i-1, j), (i, j-1), (i+1, j), (i, j+1):
            if flag[k, l] is not KNOWN:
                if flag[k, l] is INSIDE:
                    inpaint_point(k, l, image, flag, T, epsilon)
                    flag[k, l] = BAND
                    #below indent-1
                    T[k, l] = min(eikonal_solve(k-1, l, k, l-1, flag, T),
                                  eikonal_solve(k+1, l, k, l-1, flag, T),
                                  eikonal_solve(k-1, l, k, l+1, flag, T),
                                  eikonal_solve(k+1, l, k, l+1, flag, T))
                    heappush(_heap.HeapElem(T[k, l], (k, l)))

        flag[i, j] = KNOWN
        T[i, j] = -T[i, j]


def eikonal_solve(i1, j1, i2, j2, flag, T):
    """This function provides the solution for the Eikonal equation.
    """

    sol = 1.0e6
    t11 = T[i1, j1]
    t22 = T[i2, j2]
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
