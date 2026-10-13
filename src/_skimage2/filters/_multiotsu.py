#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from __future__ import annotations

try:
    import cython
except ImportError:
    exec(
        """
class cython:
    compiled = False
    class _nogil:
        def __call__(self, f):
            return f
        def __enter__(self):
            pass
        def __exit__(self, *ignore):
            pass
    nogil = _nogil()
    class _exceptval:
        def __call__(self, **kwds):
            return lambda f: f
    exceptval = _exceptval()
    @staticmethod
    def cfunc(f):
        return f
        """, globals())


import numpy as np

if cython.compiled:
    from cython.cimports import numpy as cnp
    cnp.import_array()

def _get_multiotsu_thresh_indices_lut(prob: cnp.float32_t[::1],
                                      thresh_count: cython.Py_ssize_t):
    """Finds the indices of Otsu thresholds according to the values
    occurrence probabilities.

    This implementation uses a LUT to reduce the number of floating
    point operations (see [1]_). The use of the LUT reduces the
    computation time at the price of more memory consumption.

    Parameters
    ----------
    prob : array
        Value occurrence probabilities.
    thresh_count : int
        The desired number of thresholds (classes-1).


    Returns
    -------
    py_thresh_indices : ndarray
        The indices of the desired thresholds.

    References
    ----------
    .. [1] Liao, P-S., Chen, T-S. and Chung, P-C., "A fast algorithm for
           multilevel thresholding", Journal of Information Science and
           Engineering 17 (5): 713-727, 2001. Available at:
           <https://ftp.iis.sinica.edu.tw/JISE/2001/200109_01.pdf>
           :DOI:`10.6688/JISE.2001.17.5.1`

    """

    nbins: cython.Py_ssize_t = prob.shape[0]
    py_thresh_indices = np.empty(thresh_count, dtype=np.intp)
    thresh_indices: cython.Py_ssize_t[::1] = py_thresh_indices
    current_indices: cython.Py_ssize_t[::1] = np.empty(thresh_count, dtype=np.intp)
    var_btwcls: cnp.float32_t [::1] = np.zeros((nbins * (nbins + 1)) // 2,
                                                   dtype=np.float32)
    zeroth_moment: cnp.float32_t [::1] = np.empty(nbins, dtype=np.float32)
    first_moment: cnp.float32_t [::1] = np.empty(nbins, dtype=np.float32)

    with cython.nogil:
        _set_var_btwcls_lut(prob, nbins, var_btwcls, zeroth_moment,
                            first_moment)
        _set_thresh_indices_lut(var_btwcls, hist_idx=0,
                                thresh_idx=0, nbins=nbins,
                                thresh_count=thresh_count, sigma_max=0,
                                current_indices=current_indices,
                                thresh_indices=thresh_indices)

    return py_thresh_indices


@cython.cfunc
@cython.exceptval(check=False)
@cython.nogil
def _set_var_btwcls_lut(prob: cnp.float32_t [::1],
                        nbins: cython.Py_ssize_t,
                        var_btwcls: cnp.float32_t [::1],
                        zeroth_moment: cnp.float32_t [::1],
                        first_moment: cnp.float32_t [::1]) -> cython.void:
    """Builds the lookup table containing the variance between classes.

    The variance between classes are stored in
    ``var_btwcls``. ``zeroth_moment`` and ``first_moment`` are buffers
    for storing the first row of respectively the zeroth and first
    order moments lookup table (respectively H, P and S in [1]_).

    Parameters
    ----------
    prob : array
        Value occurrence probabilities.
    nbins : int
        The number of intensity values.
    var_btwcls : array
        The upper triangular part of the lookup table containing the
        variance between classes (referred to as H in [1]_). Its size
        is equal to nbins*(nbins + 1) / 2.
    zeroth_moment : array
        First row of the zeroth order moments LUT (referred to as P in
        [1]_).
    first_moment : array
        First row of the first order moments LUT (referred to as S in
        [1]_).

    Notes
    -----
    Only the first rows of the moments lookup tables are necessary to
    build the lookup table containing the variance between
    classes. ``var_btwcls`` is stored in the compressed upper
    triangular matrix form (i.e. the seros of the lower triangular
    part are not stored).

    References
    ----------
    .. [1] Liao, P-S., Chen, T-S. and Chung, P-C., "A fast algorithm for
           multilevel thresholding", Journal of Information Science and
           Engineering 17 (5): 713-727, 2001. Available at:
           <https://ftp.iis.sinica.edu.tw/JISE/2001/200109_01.pdf>
           :DOI:`10.6688/JISE.2001.17.5.1`
    """
    i: cnp.intp_t
    j: cnp.intp_t
    idx: cnp.intp_t
    zeroth_moment_ij: cnp.float32_t
    first_moment_ij: cnp.float32_t

    zeroth_moment[0] = prob[0]
    first_moment[0] = prob[0]
    for i in range(1, nbins):
        zeroth_moment[i] = zeroth_moment[i - 1] + prob[i]
        first_moment[i] = first_moment[i - 1] + i * prob[i]
        if zeroth_moment[i] > 0:
            var_btwcls[i] = (first_moment[i]**2) / zeroth_moment[i]

    idx = nbins

    for i in range(1, nbins):
        for j in range(i, nbins):
            zeroth_moment_ij = zeroth_moment[j] - zeroth_moment[i - 1]
            if zeroth_moment_ij > 0:
                first_moment_ij = first_moment[j] - first_moment[i - 1]
                var_btwcls[idx] = (first_moment_ij**2) / zeroth_moment_ij
            idx += 1


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def _get_var_btwclas_lut(var_btwcls: cnp.float32_t [::1],
                         i: cython.Py_ssize_t, j: cython.Py_ssize_t,
                         nbins: cython.Py_ssize_t) -> cnp.float32_t:
    """Returns the variance between classes stored in compressed upper
    triangular matrix form at the desired 2D indices.

    Parameters
    ----------
    var_btwcls : array
        The lookup table containing the variance between classes in
        compressed upper triangular matrix form.
    i, j : int
        2D indices in the uncompressed lookup table.
    nbins : int
        The number of columns in the lookup table.

    Returns
    -------
    value : float
        The value of the lookup table corresponding to index (i, j).
    """
    idx: cnp.intp_t = (i * (2 * nbins - i + 1)) // 2 + j - i
    return var_btwcls[idx]


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def _set_thresh_indices_lut(
        var_btwcls: cnp.float32_t[::1], hist_idx: cython.Py_ssize_t,
        thresh_idx: cython.Py_ssize_t, nbins: cython.Py_ssize_t, thresh_count: cython.Py_ssize_t,
        sigma_max: cnp.float32_t, current_indices: cython.Py_ssize_t[::1],
        thresh_indices: cython.Py_ssize_t[::1]) -> cnp.float32_t:
    """Recursive function for finding the indices of the thresholds
    maximizing the  variance between classes sigma.

    This implementation use a lookup table of variance between classes
    to perform a brute force evaluation of sigma over all the
    combinations of threshold to find the indices maximizing sigma
    (see [1]_).

    Parameters
    ----------
    var_btwcls : array
        The upper triangular part of the lookup table containing the
        variance between classes (referred to as H in [1]_). Its size
        is equal to nbins*(nbins + 1) / 2.
    hist_idx  : int
        Current index in the histogram.
    thresh_idx : int
        Current index in thresh_indices.
    nbins : int
        number of bins used in the histogram
    thresh_count : int
        The desired number of thresholds (classes-1).
    sigma_max : float
        Current maximum variance between classes.
    current_indices : array
        Current evaluated threshold indices.
    thresh_indices : array
        The indices of thresholds maximizing the variance between
        classes.

    Returns
    -------
    max_sigma : float
        Maximum variance between classes.

    Notes
    -----
    For any candidate current_indices {t_0, ..., t_n}, sigma equals
    var_btwcls[0, t_0] + var_btwcls[t_0+1, t_1] + ...
    + var_btwcls[t_(i-1) + 1, t_i] + ... + var_btwcls[t_n, nbins-1]

    References
    ----------
    .. [1] Liao, P-S., Chen, T-S. and Chung, P-C., "A fast algorithm for
           multilevel thresholding", Journal of Information Science and
           Engineering 17 (5): 713-727, 2001. Available at:
           <https://ftp.iis.sinica.edu.tw/JISE/2001/200109_01.pdf>
           :DOI:`10.6688/JISE.2001.17.5.1`
    """
    idx: cnp.intp_t
    sigma: cnp.float32_t

    if thresh_idx < thresh_count:

        for idx in range(hist_idx, nbins - thresh_count + thresh_idx):
            current_indices[thresh_idx] = idx
            sigma_max = _set_thresh_indices_lut(var_btwcls, hist_idx=idx + 1,
                                                thresh_idx=thresh_idx + 1,
                                                nbins=nbins,
                                                thresh_count=thresh_count,
                                                sigma_max=sigma_max,
                                                current_indices=current_indices,
                                                thresh_indices=thresh_indices)

    else:

        sigma = (_get_var_btwclas_lut(var_btwcls, 0, current_indices[0], nbins)
                 + _get_var_btwclas_lut(var_btwcls,
                                        current_indices[thresh_count - 1] + 1,
                                        nbins - 1, nbins))
        for idx in range(thresh_count - 1):
            sigma += _get_var_btwclas_lut(var_btwcls, current_indices[idx] + 1,
                                          current_indices[idx + 1], nbins)

        if sigma > sigma_max:
            sigma_max = sigma
            thresh_indices[:] = current_indices[:]

    return sigma_max


def _get_multiotsu_thresh_indices(prob: cnp.float32_t [::1],
                                  thresh_count: cython.Py_ssize_t):
    """Finds the indices of Otsu thresholds according to the values
    occurrence probabilities.

    This implementation, as opposed to `_get_multiotsu_thresh_indices_lut`,
    does not use LUT. It is therefore slower.

    Parameters
    ----------
    prob : array
        Value occurrence probabilities.
    thresh_count : int
        The desired number of threshold.

    Returns
    -------
    py_thresh_indices : array
        The indices of the desired thresholds.

    """

    nbins: cython.Py_ssize_t = prob.shape[0]
    py_thresh_indices = np.empty(thresh_count, dtype=np.intp)
    thresh_indices: cython.Py_ssize_t[::1] = py_thresh_indices
    current_indices: cython.Py_ssize_t[::1] = np.empty(thresh_count, dtype=np.intp)
    zeroth_moment: cnp.float32_t [::1] = np.empty(nbins, dtype=np.float32)
    first_moment: cnp.float32_t [::1] = np.empty(nbins, dtype=np.float32)

    with cython.nogil:
        _set_moments_lut_first_row(prob, nbins, zeroth_moment, first_moment)
        _set_thresh_indices(zeroth_moment, first_moment, hist_idx=0,
                            thresh_idx=0, nbins=nbins,
                            thresh_count=thresh_count, sigma_max=0,
                            current_indices=current_indices,
                            thresh_indices=thresh_indices)

    return py_thresh_indices



@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def _set_moments_lut_first_row(prob: cnp.float32_t [::1] ,
                               nbins: cython.Py_ssize_t,
                               zeroth_moment: cnp.float32_t [::1],
                               first_moment: cnp.float32_t [::1]) -> cython.void:
    """Builds the first rows of the zeroth and first moments lookup table
    necessary to the computation of the variance between class.

    Parameters
    ----------
    prob : array
        Value occurrence probabilities.
    nbins : int
        The number of intensity values.
    zeroth_moment : array
        First row of the zeroth order moments LUT (referred to as P in
        [1]_).
    first_moment : array
        First row of the first order moments LUT (referred to as S in
        [1]_).

    References
    ----------
    .. [1] Liao, P-S., Chen, T-S. and Chung, P-C., "A fast algorithm for
           multilevel thresholding", Journal of Information Science and
           Engineering 17 (5): 713-727, 2001. Available at:
           <https://ftp.iis.sinica.edu.tw/JISE/2001/200109_01.pdf>
           :DOI:`10.6688/JISE.2001.17.5.1`
    """
    i: cnp.intp_t

    zeroth_moment[0] = prob[0]
    first_moment[0] = prob[0]
    for i in range(1, nbins):
        zeroth_moment[i] = zeroth_moment[i - 1] + prob[i]
        first_moment[i] = first_moment[i - 1] + i * prob[i]


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def _get_var_btwclas(zeroth_moment: cnp.float32_t [::1],
                     first_moment: cnp.float32_t [::1],
                     i: cython.Py_ssize_t, j: cython.Py_ssize_t) -> cnp.float32_t:
    """Computes the variance between two classes.

    Parameters
    ----------
    zeroth_moment : array
        First row of the zeroth order moments LUT (referred to as P in
        [1]_).
    first_moment : array
        First row of the first order moments LUT (referred to as S in
        [1]_).
    i, j : int
        The indices of the two considered classes.

    Returns
    -------
    value : float
        The variance between the classes i and j.
    """

    zeroth_moment_ij: cnp.float32_t
    first_moment_ij: cnp.float32_t

    if i == 0:
        if zeroth_moment[i] > 0:
            return (first_moment[j]**2) / zeroth_moment[j]
    else:
        zeroth_moment_ij = zeroth_moment[j] - zeroth_moment[i - 1]
        if zeroth_moment_ij > 0:
            first_moment_ij = first_moment[j] - first_moment[i - 1]
            return (first_moment_ij**2) / zeroth_moment_ij
    return 0


@cython.cfunc
@cython.nogil
@cython.exceptval(check=False)
def _set_thresh_indices(zeroth_moment: cnp.float32_t[::1],
                        first_moment: cnp.float32_t[::1],
                        hist_idx: cython.Py_ssize_t,
                        thresh_idx: cython.Py_ssize_t, nbins: cython.Py_ssize_t,
                        thresh_count: cython.Py_ssize_t,
                        sigma_max: cnp.float32_t,
                        current_indices: cython.Py_ssize_t[::1],
                        thresh_indices: cython.Py_ssize_t[::1]) -> cnp.float32_t:
    """Recursive function for finding the indices of the thresholds
    maximizing the  variance between classes sigma.

    This implementation uses the first rows of the zeroth and first
    moments lookup table to compute the variance between class and
    performs a brute force evaluation of sigma over all the
    combinations of threshold to find the indices maximizing sigma
    (see [1]_)..

    Parameters
    ----------
    zeroth_moment : array
        First row of the zeroth order moments LUT (referred to as P in
        [1]_).
    first_moment : array
        First row of the first order moments LUT (referred to as S in
        [1]_).
    hist_idx : int
        Current index in the histogram.
    thresh_idx : int
        Current index in thresh_indices.
    nbins : int
        number of bins used in the histogram
    thresh_count : int
        The desired number of thresholds (classes-1).
    sigma_max : float
        Current maximum variance between classes.
    current_indices : array
        Current evaluated threshold indices.
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
           <https://ftp.iis.sinica.edu.tw/JISE/2001/200109_01.pdf>
           :DOI:`10.6688/JISE.2001.17.5.1`
    """
    idx: cnp.intp_t
    sigma: cnp.float32_t

    if thresh_idx < thresh_count:

        for idx in range(hist_idx, nbins - thresh_count + thresh_idx):
            current_indices[thresh_idx] = idx
            sigma_max = _set_thresh_indices(zeroth_moment,
                                            first_moment,
                                            hist_idx=idx + 1,
                                            thresh_idx=thresh_idx + 1,
                                            nbins=nbins,
                                            thresh_count=thresh_count,
                                            sigma_max=sigma_max,
                                            current_indices=current_indices,
                                            thresh_indices=thresh_indices)

    else:

        sigma = (_get_var_btwclas(zeroth_moment, first_moment, 0,
                                  current_indices[0])
                 + _get_var_btwclas(zeroth_moment, first_moment,
                                    current_indices[thresh_count - 1] + 1,
                                    nbins - 1))
        for idx in range(thresh_count - 1):
            sigma += _get_var_btwclas(zeroth_moment, first_moment,
                                      current_indices[idx] + 1,
                                      current_indices[idx + 1])
        if sigma > sigma_max:
            sigma_max = sigma
            thresh_indices[:] = current_indices[:]

    return sigma_max
