#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

cimport numpy as cnp
cimport cython


def _get_multiotsu_thresh_indices(double [::1] prob, Py_ssize_t thresh_count,
                                  Py_ssize_t nbins):

    py_thresh_indices = np.empty(thresh_count, dtype=np.intp)
    cdef Py_ssize_t[::1] thresh_indices = py_thresh_indices
    cdef Py_ssize_t[::1] current_indices = np.empty(thresh_count, dtype=np.intp)
    cdef double [:, ::1] H = np.zeros((nbins, nbins))
    cdef double [::1] P = np.empty(nbins)
    cdef double [::1] S = np.empty(nbins)

    with nogil:
        _build_var_btwcls(prob, nbins, H, P, S)
        _set_thresh_indices(H, hist_idx=0,
                            thresh_idx=0, nbins=nbins,
                            thresh_count=thresh_count, sigma_max=0,
                            current_indices=current_indices,
                            thresh_indices=thresh_indices)

    return py_thresh_indices


cdef double [:, ::1] _build_var_btwcls(double [::1] prob,
                                       Py_ssize_t nbins,
                                       double [:, ::1] H,
                                       double [::1] P,
                                       double [::1] S) nogil:
    """Between classes variance lookup table.

    The between classes variance are stored in momS.

    Parameters
    ----------
    momP: 2D array
        Classes zeroth order moments lookup table.
    momS: 2D array
        Classes first order moments lookup table.
    bins: int
        Number of bins used in the histogram.

    """
    cdef cnp.intp_t i, j
    cdef double Pij, Sij

    P[0] = prob[0]
    S[0] = prob[0]
    for i in range(1, nbins):
        P[i] = P[i-1] + prob[i]
        S[i] = S[i-1] + i*prob[i]
        if P[i] > 0:
            H[0, i] = (S[i]**2)/P[i]

    for i in range(1, nbins):
        for j in range(i, nbins):
            Pij = P[j] - P[i-1]
            Sij = S[j] - S[i-1]
            if Pij > 0:
                H[i, j] = (Sij**2)/Pij

    return H


cdef double _set_thresh_indices(double[:, ::1] H, Py_ssize_t hist_idx,
                                Py_ssize_t thresh_idx, Py_ssize_t nbins,
                                cnp.intp_t thresh_count, double sigma_max,
                                Py_ssize_t[::1] current_indices,
                                Py_ssize_t[::1] thresh_indices) nogil:
    """
    Recursive function for calculating max_sigma.

    Parameters
    ----------
    var_btwcls : 2-d array
        Array of variance between classes.
    min_val : int
        Minimum value of the checked intervals.
    max_val : int
        Maximum value of the checked intervals.
    idx_tuple : array
        number of bins used in the histogram
    divisions : int
        Number of divisions required to generate the desired classes.
    depth : int
        Controls the iterations the algorithm had, expanding the interval
        when _find_best_rec() is called.
    max_sigma : float
        Maximum variance between classes.
    aux_thresh : array
        Values for multi-Otsu threshold.

    Returns
    -------
    max_sigma : float
        Maximum variance between classes.
    """
    cdef cnp.intp_t idx
    cdef double sigma

    if thresh_idx < thresh_count:

        for idx in range(hist_idx, nbins-thresh_count+thresh_idx):
            current_indices[thresh_idx] = idx
            sigma_max = _set_thresh_indices(H, hist_idx=idx+1,
                                            thresh_idx=thresh_idx+1,
                                            nbins=nbins,
                                            thresh_count=thresh_count,
                                            sigma_max=sigma_max,
                                            current_indices=current_indices,
                                            thresh_indices=thresh_indices)

    else:

        sigma = (H[0, current_indices[0]]
                 + H[current_indices[thresh_count-1]+1, nbins-1])
        for idx in range(thresh_count-1):
            sigma += H[current_indices[idx]+1, current_indices[idx+1]]
        if sigma > sigma_max:
            sigma_max = sigma
            thresh_indices[:] = current_indices[:]

    return sigma_max
