cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def _line_nd_cy(
        Py_ssize_t[::1] cur,
        Py_ssize_t[::1] step,
        Py_ssize_t[::1] delta,
        *,
        Py_ssize_t x_dim,
        Py_ssize_t[::1] error,
        Py_ssize_t[:, ::1] coords
):

    cdef int n_dim = error.shape[0]
    cdef int n_points = coords.shape[1]
    cdef int i_pt
    for i_pt in range(n_points):
        for i_dim in range(n_dim):
            coords[i_dim, i_pt] = cur[i_dim]
            if error[i_dim] > 0:
                coords[i_dim, i_pt] += step[i_dim]
                error[i_dim] -= 2 * delta[x_dim]
            error[i_dim] += 2 * delta[i_dim]
            cur[i_dim] = coords[i_dim, i_pt]
