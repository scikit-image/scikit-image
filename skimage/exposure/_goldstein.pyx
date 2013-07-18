#cython: cdivision=True
#cython: boundscheck=True
#cython: nonecheck=False
#cython: wraparound=True

from __future__ import print_function

import numpy as np

from libc.math cimport M_PI, lround
cimport numpy as cnp


cdef struct branch_cut:
    # TODO: clearer definitions of where the cuts lie
    # Positions of the cuts are relative to the pixel in the phase image
    # Using uint8 would be natural here, but gives the error
    # "ValueError: Buffer dtype mismatch; next field is at offset 2 but
    # 4 expected"
    cnp.uint16_t vcut    # cut normal to 1st dimension
    cnp.uint16_t hcut    # cut normal to 0th dimension
    # The residue resides in the bottom right corner of the pixel in the
    # phase image; the cut edges in the struct are thus _above_ and _left_ of
    # the residue
    cnp.uint32_t residue_no  # index into an array of residues


branch_cut_dtype = np.dtype([('vcut', np.uint16), ('hcut', np.uint16),
                             ('residue_no', np.uint32)])


cdef inline double _phase_difference(double from_, double to):
    cdef double d = to - from_
    if d > M_PI:
        d -= 2 * M_PI
    elif d < -M_PI:
        d += 2 * M_PI
    return d


def find_phase_residues_cy(double[:, ::1] image):
    residues_array = np.zeros((image.shape[0], image.shape[1]),
                              dtype=np.int8, order='C')
    cdef:
        cnp.int8_t[:, ::1] residues = residues_array
        Py_ssize_t i, j
        double s
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            s = (_phase_difference(image[i - 1, j - 1], image[i - 1, j])
                 + _phase_difference(image[i - 1, j], image[i, j])
                 + _phase_difference(image[i, j], image[i, j - 1])
                 + _phase_difference(image[i, j - 1], image[i - 1, j - 1]))
            residues[i, j] = lround(s / (2 * M_PI))
    return residues_array


def _prepare_branch_cuts(branch_cut[:, ::1] branch_cuts,
                         cnp.int_t[::1] residue_storage, Py_ssize_t index,
                         cnp.int8_t[:, ::1] residues,
                         cnp.uint8_t[:, ::1] mask):
    '''Prepare the branch_cuts structures for branch cut finding.

    Parameters
    ----------
    branch_cuts : output parameter, will be modified in-place
        Branch cut struct
    residue_storage : output parameter, will be modified in-place
        Storage for residues
    index :
        Element of ``residue_storage`` where the first residue should be stored
    residues :
        Residues as determined by find_phase_residues
    mask :
        Mask of the ``residues`` array; masked entries will not be stored.
    '''
    cdef:
        Py_ssize_t i, j
    for i in range(residues.shape[0]):
        for j in range(residues.shape[1]):
            if residues[i, j] != 0 and mask[i, j] == 0:
                # Found an unmasked residue
                residue_storage[index] = residues[i, j]
                #branch_cuts[i, j]['residue_no'] = index
                branch_cuts[i, j].residue_no = index
                index += 1
    return None
