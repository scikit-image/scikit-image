# distutils: language = c++

from libcpp.unordered_map cimport unordered_map
cimport cython
from .._shared.fused_numerics cimport np_numeric, np_anyint

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing
def _map_array(np_anyint[:] inarr, np_numeric[:] outarr,
               np_anyint[:] inval, np_numeric[:] outval):
    # build the map from the input and output vectors
    cdef size_t i, n_map, n_array
    cdef unordered_map[np_anyint, np_numeric] lut
    n_map = inval.shape[0]
    for i in range(n_map):
        lut[inval[i]] = outval[i]
    # apply the map to the array
    n_array = inarr.shape[0]
    # The prange option gave some compilation warnings
    #  "Unsigned index type not allowed before OpenMP 3.0"
    # and didn't seem to be any faster
    # for i in prange(n_array, nogil=True): #
    for i in range(n_array):
        outarr[i] = lut[inarr[i]]
