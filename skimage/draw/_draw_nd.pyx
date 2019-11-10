import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def my_line_nd_cython(long[::1] start, long[::1] stop, bint endpoint=False, bint integer=True):
    # TODO currently ignoring the decimal part of the number
    if stop.shape[0] != start.shape[0]:
        raise ValueError("Different lengths")
    cdef int n = stop.shape[0]
    cdef int[::1] delta = np.empty(n, dtype=np.intc)
    cdef int[::1] steps = np.empty(n, dtype=np.intc)
    cdef int n_points = 0
    cdef int x_dim = -1
    cdef int i_dim
    for i_dim in range(n):
        if stop[i_dim] >= start[i_dim]:
            delta[i_dim] = stop[i_dim] - start[i_dim]
            steps[i_dim] = +1
        else:
            delta[i_dim] = start[i_dim] - stop[i_dim]
            steps[i_dim] = -1
        
        if abs(delta[i_dim]) > n_points:
            n_points = delta[i_dim]
            x_dim = i_dim
    
    if endpoint:
        n_points += 1

    cdef int[::1] cum_error = np.empty(n, dtype=np.intc)
    cdef int[:, ::1] coords = np.empty([n_points, n], dtype=np.intc)
    
    for i_dim in range(n):
        cum_error[i_dim] = 4 * delta[i_dim] - delta[x_dim]
        coords[0, i_dim] = start[i_dim]
    
    cdef int i_pt
    for i_pt in range(1, n_points):
        for i_dim in range(n):
            coords[i_pt, i_dim] = coords[i_pt-1, i_dim]
            if cum_error[i_dim] > 0:
                coords[i_pt, i_dim] += steps[i_dim]
                cum_error[i_dim] -= 2 * delta[x_dim]
            cum_error[i_dim] += 2 * delta[i_dim]
        
    return np.asarray(coords)