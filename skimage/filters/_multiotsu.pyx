#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

cimport numpy as cnp
cimport cython


def _get_multiotsu_thresh_indices_lut(double [::1] prob,
                                      Py_ssize_t thresh_count):

    cdef Py_ssize_t nbins = prob.shape[0]
    py_thresh_indices = np.empty(thresh_count, dtype=np.intp)
    cdef Py_ssize_t[::1] thresh_indices = py_thresh_indices
    cdef Py_ssize_t[::1] current_indices = np.empty(thresh_count, dtype=np.intp)
    cdef double [:, ::1] H = np.zeros((nbins, nbins))
    cdef double [::1] P = np.empty(nbins)
    cdef double [::1] S = np.empty(nbins)

    with nogil:
        _set_var_btwcls(prob, nbins, H, P, S)
        _set_thresh_indices_lut(H, hist_idx=0,
                                thresh_idx=0, nbins=nbins,
                                thresh_count=thresh_count, sigma_max=0,
                                current_indices=current_indices,
                                thresh_indices=thresh_indices)

    return py_thresh_indices


def _get_multiotsu_thresh_indices(double [::1] prob, Py_ssize_t thresh_count):

    cdef Py_ssize_t nbins = prob.shape[0]
    py_thresh_indices = np.empty(thresh_count, dtype=np.intp)
    cdef Py_ssize_t[::1] thresh_indices = py_thresh_indices
    cdef Py_ssize_t[::1] current_indices = np.empty(thresh_count, dtype=np.intp)
    cdef double [:, ::1] H = np.zeros((nbins, nbins))
    cdef double [::1] P = np.empty(nbins)
    cdef double [::1] S = np.empty(nbins)

    with nogil:
        _set_moment_lut_first_row(prob, nbins, P, S)
        _set_thresh_indices(P, S, hist_idx=0,
                            thresh_idx=0, nbins=nbins,
                            thresh_count=thresh_count, sigma_max=0,
                            current_indices=current_indices,
                            thresh_indices=thresh_indices)

    return py_thresh_indices


cdef void _set_moment_lut_first_row(double [::1] prob,
                                    Py_ssize_t nbins,
                                    double [::1] P,
                                    double [::1] S) nogil:
    cdef cnp.intp_t i

    P[0] = prob[0]
    S[0] = prob[0]
    for i in range(1, nbins):
        P[i] = P[i-1] + prob[i]
        S[i] = S[i-1] + i*prob[i]


cdef double _get_var_btwclas(double [::1] P, double [::1] S,
                             Py_ssize_t i, Py_ssize_t j) nogil:

    cdef double Pij, Sij

    if i == 0:
        if P[i] > 0:
            return (S[j]**2)/P[j]
    else:
        Pij = P[j] - P[i-1]
        if Pij > 0:
            Sij = S[j] - S[i-1]
            return (Sij**2)/Pij
    return 0


cdef double _set_thresh_indices(double[::1] P, double[::1] S,
                                Py_ssize_t hist_idx,
                                Py_ssize_t thresh_idx, Py_ssize_t nbins,
                                Py_ssize_t thresh_count, double sigma_max,
                                Py_ssize_t[::1] current_indices,
                                Py_ssize_t[::1] thresh_indices) nogil:
    """Recursive function for finding the indices of the thresholds
    maximizing the  variance between classes sigma.

     This implementation is brute force evaluation of sigma over all
     the combinations of threshold to find the indices maximizing
     sigma (see [1]).

    Parameters
    ----------
    H : (nbins, nbins) 2D array
        Lookup table of variance between classes.
    hist_idx : int
        Current index in the histogram.
    thresh_idx : int
        Current index in thresh_indices.
    nbins : int
        number of bins used in the histogram
    thresh_count : int
        Number of divisions required to generate the desired classes.
    sigma_max : float
        Current maximum variance between classes.
    current_indices: (thresh_count, ) 1D array
        Current evalueted threshold indices.
    thresh_indices : array
        The indices of thresholds maximizing the variance between
        classes.

    Returns
    -------
    max_sigma : float
        Maximum variance between classes.

    References
    ----------
    .. [1] Liao, P-S., Chen, T-S. and Chung, P-C., "A fast algorithm for
           multilevel thresholding", Journal of Information Science and
           Engineering 17 (5): 713-727, 2001. Available at:
           <http://ftp.iis.sinica.edu.tw/JISE/2001/200109_01.pdf>

    """
    cdef cnp.intp_t idx
    cdef double sigma

    if thresh_idx < thresh_count:

        for idx in range(hist_idx, nbins-thresh_count+thresh_idx):
            current_indices[thresh_idx] = idx
            sigma_max = _set_thresh_indices(P, S, hist_idx=idx+1,
                                            thresh_idx=thresh_idx+1,
                                            nbins=nbins,
                                            thresh_count=thresh_count,
                                            sigma_max=sigma_max,
                                            current_indices=current_indices,
                                            thresh_indices=thresh_indices)

    else:

        sigma = (_get_var_btwclas(P, S, 0, current_indices[0])
                 + _get_var_btwclas(P, S,
                                    current_indices[thresh_count-1]+1,
                                    nbins-1))
        for idx in range(thresh_count-1):
            sigma += _get_var_btwclas(P, S, current_indices[idx]+1,
                                      current_indices[idx+1])
        if sigma > sigma_max:
            sigma_max = sigma
            thresh_indices[:] = current_indices[:]

    return sigma_max


cdef void _set_var_btwcls(double [::1] prob,
                          Py_ssize_t nbins,
                          double [:, ::1] H,
                          double [::1] P,
                          double [::1] S) nogil:
    """Build the between classes variance lookup table.

    The between classes variance are stored in H. P and S are buffers
    for storing the first row of respectively the zeroth and first
    order moments lookup table (see [1]).

    Parameters
    ----------
    prob: 2D array
        Intensities probabilies.
    nbins: int
        The number of intensity values.
    H: (nbins, nbins) 2D array
        The between classes variance lookup table.
    P: (nbins, ) 1D array
        First row of the zeroth order moments LUT (see [1]).
    S: (nbins, ) 1D array
        First row of the first order moments LUT (see [1]).

    References
    ----------
    .. [1] Liao, P-S., Chen, T-S. and Chung, P-C., "A fast algorithm for
           multilevel thresholding", Journal of Information Science and
           Engineering 17 (5): 713-727, 2001. Available at:
           <http://ftp.iis.sinica.edu.tw/JISE/2001/200109_01.pdf>

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
            if Pij > 0:
                Sij = S[j] - S[i-1]
                H[i, j] = (Sij**2)/Pij


cdef double _set_thresh_indices_lut(double[:, ::1] H, Py_ssize_t hist_idx,
                                    Py_ssize_t thresh_idx, Py_ssize_t nbins,
                                    Py_ssize_t thresh_count, double sigma_max,
                                    Py_ssize_t[::1] current_indices,
                                    Py_ssize_t[::1] thresh_indices) nogil:
    """Recursive function for finding the indices of the thresholds
    maximizing the  variance between classes sigma.

     This implementation is brute force evaluation of sigma over all
     the combinations of threshold to find the indices maximizing
     sigma (see [1]).

    For any candidate current_indices {t_0, ..., t_n}, sigma equals
    H[0, t_0] + H[t_0+1, t_1] + ... + H[t_(i-1) + 1, t_i] + ... H[t_n, nbins-1]

    Parameters
    ----------
    H : (nbins, nbins) 2D array
        Lookup table of variance between classes.
    hist_idx : int
        Current index in the histogram.
    thresh_idx : int
        Current index in thresh_indices.
    nbins : int
        number of bins used in the histogram
    thresh_count : int
        Number of divisions required to generate the desired classes.
    sigma_max : float
        Current maximum variance between classes.
    current_indices: (thresh_count, ) 1D array
        Current evalueted threshold indices.
    thresh_indices : array
        The indices of thresholds maximizing the variance between
        classes.

    Returns
    -------
    max_sigma : float
        Maximum variance between classes.

    References
    ----------
    .. [1] Liao, P-S., Chen, T-S. and Chung, P-C., "A fast algorithm for
           multilevel thresholding", Journal of Information Science and
           Engineering 17 (5): 713-727, 2001. Available at:
           <http://ftp.iis.sinica.edu.tw/JISE/2001/200109_01.pdf>

    """
    cdef cnp.intp_t idx
    cdef double sigma

    if thresh_idx < thresh_count:

        for idx in range(hist_idx, nbins-thresh_count+thresh_idx):
            current_indices[thresh_idx] = idx
            sigma_max = _set_thresh_indices_lut(H, hist_idx=idx+1,
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
