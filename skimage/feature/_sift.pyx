#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np

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


cpdef _local_max(np_floats[:, :, ::1] octave, double thresh):
    cdef:
        Py_ssize_t n_r = octave.shape[0]
        Py_ssize_t n_c = octave.shape[1]
        Py_ssize_t n_s = octave.shape[2]
        Py_ssize_t r, c, s
        np_floats* center_ptr
        Py_ssize_t neighbor_offsets[26]
        np_floats center_val, val
        int n = 0

    for r in range(-1, 2):
        for c in range(-1, 2):
            for s in range(-1, 2):
                if (r != 0 or c != 0 or s != 0):
                    neighbor_offsets[n] = (r * n_c + c) * n_s + s
                    n += 1

    maxima_coords = []
    for r in range(1, n_r - 1):
        for c in range(1, n_c - 1):
            for s in range(1, n_s - 1):
                center_ptr = &octave[r, c, s]
                center_val = center_ptr[0]
                if abs(center_val) < thresh:
                    continue
                is_local_min = True
                for n in range(26):
                    val = center_ptr[neighbor_offsets[n]]
                    if val <= center_val:
                        is_local_min = False
                        break

                if is_local_min:
                    is_local_max = False
                else:
                    is_local_max = True
                    for n in range(26):
                        val = center_ptr[neighbor_offsets[n]]
                        if val >= center_val:
                            is_local_max = False
                            break
                if is_local_min or is_local_max:
                    maxima_coords.append((r, c, s))
    return np.asarray(maxima_coords, dtype=np.intp)
