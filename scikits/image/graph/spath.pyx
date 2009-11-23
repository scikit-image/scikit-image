# -*- python -*-

import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double fabs(double f)

cpdef shortest_path(np.ndarray arr, int reach=1):
     """Find the shortest left-to-right path through an array.

     Parameters
     ----------
     arr : (M, N) ndarray of float64
     reach : int, optional
         By default (``reach = 1``), the shortest path can only move
         one row up or down for every column it moves forward (i.e.,
         the path gradient is limited to 1).  `reach` defines the
         number of rows that can be skipped at each step.

     Returns
     -------
     p : ndarray of int
         For each column, give the row-coordinate of the
         shortest path.
     cost : float
         Cost of path.  This is the absolute sum of all the
         differences along the path.

     """
     if arr.ndim != 2:
          raise ValueError("Expected 2-D array as input")

     cdef np.ndarray[np.double_t, ndim=2] data = \
          np.ascontiguousarray(arr, dtype=np.double)

     cdef int M = arr.shape[0]
     cdef int N = arr.shape[1]

     cdef np.ndarray[np.int_t, ndim=2] node = \
          np.empty((M, N), dtype=int)

     cdef np.ndarray[np.double_t, ndim=2] cost = \
          np.empty((M, N), dtype=np.double)

     cdef np.ndarray[np.int_t] out = np.empty((N,), dtype=int)

     cdef int c, r, rb, r_min_node
     cdef int r_bracket_min = 0, r_bracket_max = 0
     cdef double delta0 = 0, delta1 = 0

     cost[:, 0] = 0

     for c in range(1, N):
          for r in range(M):
               r_bracket_min = r - reach
               r_bracket_max = r + reach

               if r_bracket_min < 0:
                    r_bracket_min = 0
               if r_bracket_max > M - 1:
                    r_bracket_max = M - 1

               node[r, c] = r_bracket_min
               for rb in range(r_bracket_min, r_bracket_max + 1):
                    delta0 = fabs(data[r, c] - data[rb, c - 1])
                    delta1 = fabs(data[r, c] - data[node[r, c], c - 1])
                    if delta0 < delta1:
                         node[r, c] = rb

               cost[r, c] = cost[node[r, c], c - 1] + \
                            fabs(data[r, c] - data[node[r, c], c - 1])

     # Find minimum cost path
     r_min_node = cost[:,-1].argmin()

     # Backtrack
     out[N - 1] = r_min_node
     for c in range(N - 1, 0, -1):
          out[c - 1] = node[out[c], c]

     return out, cost[r_min_node, N - 1]
