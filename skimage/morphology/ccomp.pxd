"""Export fast union find in Cython"""
cimport numpy as np

DTYPE = np.int
ctypedef np.int_t DTYPE_t

cdef DTYPE_t find_root(np.int_t *forest, np.int_t n)
cdef set_root(np.int_t *forest, np.int_t n, np.int_t root)
cdef join_trees(np.int_t *forest, np.int_t n, np.int_t m)
cdef link_bg(np.int_t *forest, np.int_t n, np.int_t *background_node)
