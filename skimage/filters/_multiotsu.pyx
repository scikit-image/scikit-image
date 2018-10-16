#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
import numpy as np

cimport numpy as cnp
cimport cython
from itertools import combinations

def _find_threshold_multiotsu(double [:, ::1] var_btwcls, Py_ssize_t classes,
                              Py_ssize_t bins, double [::1] aux_thresh):
    """
    Cython utility function for finding thresholds in multi-Otsu algorithm.
    This function is only called by filters.threshold_multiotsu.

    Parameters
    ----------

    var_btwcls : 2-d array
        array of variance between classes
    classes ! int
        number of classes to be segmented
    bins : int
        number of bins used in the histogram
    aux_thresh : array
        thresholds to be returned
    """
    cdef Py_ssize_t idx1, idx2, idx3, idx4, idd
    cdef Py_ssize_t sh0, sh1
    cdef double part_sigma = 0
    cdef double max_sigma = 0
    cdef Py_ssize_t [:, ::1] _tmp = np.array(list(
                                combinations(range(1, bins - 2), classes - 1)))
    sh0 = _tmp.shape[0]
    sh1 = _tmp.shape[1]
    cdef Py_ssize_t [:, :] idx_tuples = np.zeros((sh0, sh1 + 2), 
                                                  dtype=np.int)
    idx_tuples[:, 1:sh1 + 1] = _tmp[:]
    idx_tuples[:, sh1 + 1] = bins - 1
    cdef Py_ssize_t [::1] idx_tuple
    for idx_tuple in idx_tuples:
        part_sigma = 0
        for idd in range(0, classes):
            part_sigma += var_btwcls[1 + idx_tuple[idd], idx_tuple[idd+1]]
        if max_sigma < part_sigma:
            for idd in range(0, classes - 1):
                aux_thresh[idd] = idx_tuple[idd + 1]
                max_sigma = part_sigma

    return aux_thresh


def _find_threshold_multiotsu_bk(double [:, ::1] var_btwcls, Py_ssize_t classes,
                              Py_ssize_t bins, double [::1] aux_thresh):
    """
    Cython utility function for finding thresholds in multi-Otsu algorithm.
    This function is only called by filters.threshold_multiotsu.

    Parameters
    ----------

    var_btwcls : 2-d array
        array of variance between classes
    classes ! int
        number of classes to be segmented
    bins : int
        number of bins used in the histogram
    aux_thresh : array
        thresholds to be returned
    """
    cdef Py_ssize_t idx1, idx2, idx3, idx4
    cdef double part_sigma = 0
    cdef double max_sigma = 0

    if classes == 2:
        for idx1 in range(1, bins - classes):
            part_sigma = var_btwcls[1, idx1] + var_btwcls[idx1+1, bins-1]
            if max_sigma < part_sigma:
                aux_thresh[0] = idx1
                max_sigma = part_sigma

    elif classes == 3:
        for idx1 in range(1, bins - classes):
            for idx2 in range(idx1+1, bins - classes+1):
                part_sigma = var_btwcls[1, idx1] + \
                            var_btwcls[idx1+1, idx2] + \
                            var_btwcls[idx2+1, bins-1]
                if max_sigma < part_sigma:
                    aux_thresh[0] = idx1
                    aux_thresh[1] = idx2
                    max_sigma = part_sigma

    elif classes == 4:
        for idx1 in range(1, bins - classes):
            for idx2 in range(idx1+1, bins - classes+1):
                for idx3 in range(idx2+1, bins - classes+2):
                    part_sigma = var_btwcls[1, idx1] + \
                                var_btwcls[idx1+1, idx2] + \
                                var_btwcls[idx2+1, idx3] + \
                                var_btwcls[idx3+1, bins-1]

                    if max_sigma < part_sigma:
                        aux_thresh[0] = idx1
                        aux_thresh[1] = idx2
                        aux_thresh[2] = idx3
                        max_sigma = part_sigma

    elif classes == 5:
        for idx1 in range(1, bins - classes):
            for idx2 in range(idx1+1, bins - classes+1):
                for idx3 in range(idx2+1, bins - classes+2):
                    for idx4 in range(idx3+1, bins - classes+3):
                        part_sigma = var_btwcls[1, idx1] + \
                            var_btwcls[idx1+1, idx2] + \
                            var_btwcls[idx2+1, idx3] + \
                            var_btwcls[idx3+1, idx4] + \
                            var_btwcls[idx4+1, bins-1]

                        if max_sigma < part_sigma:
                            aux_thresh[0] = idx1
                            aux_thresh[1] = idx2
                            aux_thresh[2] = idx3
                            aux_thresh[3] = idx4
                            max_sigma = part_sigma

    return aux_thresh
