import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def _line_nd_cy(Py_ssize_t[::1] cur, Py_ssize_t[::1] stop, Py_ssize_t[::1] step,
                *, Py_ssize_t[::1] delta, Py_ssize_t x_dim,
                bint endpoint, Py_ssize_t[::1] error, Py_ssize_t[:, ::1] coords):

    ndim = error.shape[0]
    n_points = coords.shape[1]

    cdef int i_pt
    for i_pt in range(n_points):
        for i_dim in range(ndim):
            coords[i_dim, i_pt] = cur[i_dim]
            if error[i_dim] > 0:
                coords[i_dim, i_pt] += step[i_dim]
                error[i_dim] -= 2 * delta[x_dim]
            error[i_dim] += 2 * delta[i_dim]
            cur[i_dim] = coords[i_dim, i_pt]
