#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp
from libc.math cimport M_PI, floor

from .._shared.fused_numerics cimport np_floats


cpdef _ori_distances(np_floats[::1] ori_bins,
                     np_floats[::1] theta):
    """Compute angular minima and their indices.

    Parameters
    ----------
    ori_bins : ndarray
        The orientation histogram bins.
    theta : ndarray
        The orientation computed for each pixel in the patch.

    Returns
    -------
    near_t : ndarray
        The histogram bin index corresponding to each pixel in the patch.
    near_t_val : ndarray
        The distance between the orientation of each pixel and the nearest bin.
    """
    cdef:
        Py_ssize_t n_theta = theta.size
        Py_ssize_t n_ori = ori_bins.size
        Py_ssize_t i, j, idx_min
        np_floats th, dist, dist_min
        np_floats two_pi = 2 * M_PI
        Py_ssize_t[::1] near_t = np.empty((n_theta, ), dtype=np.intp)

    if np_floats == cnp.float32_t:
        dtype = np.float32
    else:
        dtype = np.float64
    cdef np_floats[::1] near_t_vals = np.empty((n_theta, ), dtype=dtype)

    for i in range(n_theta):
        dist_min = 100.
        th = theta[i]
        for j in range(n_ori):
            dist = ori_bins[j] - th
            if dist > two_pi:
                dist -= two_pi
            elif dist < 0:
                dist += two_pi
            if dist < dist_min:
                idx_min = j
                dist_min = dist
        near_t[i] = idx_min
        near_t_vals[i] = dist_min
    return np.asarray(near_t), np.asarray(near_t_vals)


cpdef _update_histogram(np_floats[:, :, ::1] histograms,
                        const Py_ssize_t[::1] near_t,
                        np_floats[::1] near_t_val,  # const
                        np_floats[::1] magnitude,  # const
                        np_floats[:, ::1] dist_r,  # const
                        np_floats[:, ::1] dist_c,  # const
                        np_floats rc_bin_spacing):
    """Compute an array of orientation histograms (Eq. 28 of Otero et. al.)

    Parameters
    ----------
    histograms : (n_hist, n_hist, n_ori) ndarray
        An array of zeros that will contain the histogram output. `n_ori` is
        the number of orientation bins and `n_hist` is the number of spatial
        bins along each axis.
    near_t : (n_patch,) ndarray
        The orientation histogram bins obtained from `_ori_distances`.
    near_t_val : (n_patch,) ndarray
        The orientation histogram values obtained from `_ori_distances`.
    magnitude : (n_patch,) ndarray
        The magnitude weights based on spatial distance.
    dist_r : (n_hist, n_patch) ndarray
        Row distances between each point in the patch and the nearest row
        histogram bin. Shape (n_hist, n_patch).
    dist_c : (n_hist, n_patch) ndarray
        Column distances between each point in the patch and the nearest column
        histogram bin.  Shape (n_hist, n_patch).
    rc_bin_spacing : float
        The spacing between spatial histogram bins along a single axis.

    """
    cdef:
        Py_ssize_t i, p, r, c, k_index, k_index2
        Py_ssize_t n_patch = len(magnitude)
        Py_ssize_t n_hist = histograms.shape[0]
        Py_ssize_t n_ori = histograms.shape[2]
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
                w2 = w0 - w1
                k_index = near_t[p]
                k_index2 = k_index + 1
                if k_index2 == n_ori:
                    k_index2 = 0
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
                is_local_max = False
                for offset in neighbor_offsets:
                    val = center_ptr[offset]
                    if val <= center_val:
                        is_local_max = True
                        is_local_min = False
                        for offset in neighbor_offsets:
                            val = center_ptr[offset]
                            if val >= center_val:
                                is_local_max = False
                                break
                        break
                if is_local_min or is_local_max:
                    maxima_coords.append((r, c, s))
    return np.asarray(maxima_coords, dtype=np.intp)
