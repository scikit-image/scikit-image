import numpy as np
import cython

cimport numpy as cnp
from .._shared.fused_numerics cimport np_floats


 # Can't use "const np_floats[::1]", but instead must declare fused const types
 # See: https://github.com/cython/cython/issues/1772

cdef fused const_float_1d_c:
    const float[::1]
    const double[::1]

cdef fused const_floating_2d_c:
    const float[:, ::1]
    const double[:, ::1]


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.cdivision(True)  # C style integer division
cpdef Py_ssize_t _build_matrix_inner(
    # starting index offsets
    Py_ssize_t row_start,
    Py_ssize_t known_start_idx,
    Py_ssize_t unknown_start_idx,
    # input arrays
    const Py_ssize_t [::1] center_i,
    const Py_ssize_t [::1] raveled_offsets,
    const_float_1d_c coef_vals,
    const short [::1] mask_flat,
    const_floating_2d_c out_flat,
    # output arrays
    Py_ssize_t [::1] row_idx_known,
    np_floats [:, ::1] data_known,
    Py_ssize_t [::1] row_idx_unknown,
    Py_ssize_t [::1] col_idx_unknown,
    np_floats [::1] data_unknown
):
    """Fill values in *_known and *_unkown"""
    cdef:
        Py_ssize_t i, o, ch
        Py_ssize_t row_idx, known_idx, unknown_idx, loc, n_known
        Py_ssize_t num_offsets = len(raveled_offsets)
        Py_ssize_t npix = len(center_i)
        np_floats cval
        int nchannels = data_known.shape[1]

    row_idx = row_start
    known_idx = known_start_idx
    unknown_idx = unknown_start_idx
    for i in range(npix):
        n_known = 0
        for o in range(num_offsets):
            loc = center_i[i] + raveled_offsets[o]
            cval = coef_vals[o]
            # print(f"    loc={loc}")
            if mask_flat[loc]:
                data_unknown[unknown_idx] = cval
                row_idx_unknown[unknown_idx] = row_idx
                col_idx_unknown[unknown_idx] = loc
                unknown_idx += 1
            else:
                cval = coef_vals[o]
                for ch in range(nchannels):
                    data_known[known_idx, ch] -= cval * out_flat[loc, ch]
                n_known += 1
        if n_known > 0:
            row_idx_known[known_idx] = row_idx
            known_idx += 1
            # print(f"row_idx={row_idx}, unknown_idx={unknown_idx}, known_idx={known_idx}")
        row_idx += 1
    return known_idx
