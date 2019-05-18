#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

cimport numpy as cnp
cimport cython
from itertools import combinations

def _find_threshold_multiotsu(double [:, ::1] var_btwcls,
                              Py_ssize_t classes,
                              Py_ssize_t bins):
    """
    Cython utility function for finding thresholds in multi-Otsu algorithm.
    This function is only called by filters.threshold_multiotsu.

    Parameters
    ----------
    var_btwcls : 2-d array
        array of variance between classes
    classes : int
        number of classes to be segmented
    bins : int
        number of bins used in the histogram
    aux_thresh : array
        thresholds to be returned
    """
    cdef Py_ssize_t idd
    cdef Py_ssize_t sh0, sh1
    cdef double part_sigma = 0
    # max_sigma is the maximum variance between classes.
    cdef double max_sigma = 0
    #cdef Py_ssize_t [:, ::1] _tmp = np.array(list(
    #    combinations(range(1, bins - 2), classes - 1)))

#    sh0 = _tmp.shape[0]
#    sh1 = _tmp.shape[1]
#    cdef cnp.intp_t[:, ::1] idx_tuples = np.zeros((sh0, sh1 + 2),
#                                                   dtype=np.intp)
#    idx_tuples[:, 1:sh1 + 1] = _tmp[:]
#    idx_tuples[:, sh1 + 1] = bins - 1
#    cdef cnp.intp_t[::1] idx_tuple = np.zeros(classes-1, dtype=np.intp)

    py_aux_thresh = np.empty(classes - 1)
    cdef double[::1] aux_thresh = py_aux_thresh

    cdef cnp.intp_t[::1] idx_tuple = np.zeros(classes+1, dtype=np.intp)
    idx_tuple[classes] = bins - 1

    _find_best_rec(var_btwcls=var_btwcls, min_val=0, max_val=bins-1, idx_tuple=idx_tuple, divisions=classes-1, depth=0, max_sigma=0, aux_thresh=aux_thresh)

    return py_aux_thresh


cdef _find_best_rec(double[:, ::1] var_btwcls, cnp.intp_t min_val, cnp.intp_t max_val, cnp.intp_t[::1] idx_tuple, cnp.intp_t divisions, cnp.intp_t depth, double max_sigma, double[::1] aux_thresh):
    """"""
    cdef cnp.intp_t idx, idd
    cdef double part_sigma

    if divisions-1 == depth:
        for idx in range(min_val, max_val-divisions+depth):
            # starting calculations
            idx_tuple[depth+1] = idx
            part_sigma = 0
            for idd in range(divisions+1):
                part_sigma += var_btwcls[1 + idx_tuple[idd], idx_tuple[idd+1]]
                # checking if partial sigma is higher than maximum sigma
                if max_sigma < part_sigma:
                    for idd in range(divisions):
                        aux_thresh[idd] = idx_tuple[idd + 1]
                    max_sigma = part_sigma

    else:
        for idx in range(min_val, max_val-divisions+depth):
            idx_tuple[depth+1] = idx
            max_sigma = _find_best_rec(var_btwcls=var_btwcls, min_val=idx+1, max_val=max_val, idx_tuple=idx_tuple, divisions=divisions, depth=depth+1, max_sigma=max_sigma, aux_thresh=aux_thresh)

    return max_sigma
