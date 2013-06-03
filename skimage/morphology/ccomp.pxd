"""Export fast union find in Cython"""
cimport numpy as cnp

DTYPE = cnp.intp
ctypedef cnp.intp_t DTYPE_t

cdef DTYPE_t find_root(DTYPE_t *forest, DTYPE_t n)
cdef set_root(DTYPE_t *forest, DTYPE_t n, DTYPE_t root)
cdef join_trees(DTYPE_t *forest, DTYPE_t n, DTYPE_t m)
cdef link_bg(DTYPE_t *forest, DTYPE_t n, DTYPE_t *background_node)
