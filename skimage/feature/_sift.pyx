#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from libc.math cimport M_PI
from .._shared.fused_numerics cimport np_floats


cpdef _update_histogram(np_floats[:, :, ::1] histograms,
                        const Py_ssize_t[::1] near_t,
                        np_floats[::1] magnitude,  # const
                        np_floats[::1] near_t_val,  # const
                        np_floats[:, ::1] dist_r,  # const
                        np_floats[:, ::1] dist_c,  # const
                        Py_ssize_t n_patch,
                        Py_ssize_t n_hist,
                        Py_ssize_t n_ori,
                        np_floats rc_bin_spacing):
    cdef:
        Py_ssize_t i, p, r, c, k_index, k_index2
        np_floats t_val, val_norm1, w0, w1, w2, inv_spacing
    inv_spacing = 1 / rc_bin_spacing
    val_norm1 = n_ori / (2 * M_PI)
    for p in range(n_patch):
        for r in range(n_hist):
            if dist_r[r, p] > rc_bin_spacing:
                continue
            for c in range(n_hist):
                if dist_c[c, p] > rc_bin_spacing:
                    continue
                w0 = ((1 - inv_spacing * dist_r[r, p])
                      * (1 - inv_spacing * dist_c[c, p])
                      * magnitude[p])
                t_val = near_t_val[p]
                w1 = w0 * val_norm1 * t_val
                w2 = w0 * (1 - val_norm1 * t_val)
                k_index = near_t[p]
                if k_index == n_ori - 1:
                    k_index2 = 0
                else:
                    k_index2 = k_index + 1
                histograms[r, c, k_index] += w1
                histograms[r, c, k_index2] += w2
