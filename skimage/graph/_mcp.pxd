""" This is the definition file for mcp.pyx.
It contains the definitions of the mcp class, such that
other cython modules can "cimport mcp" and subclass it.
"""

cimport numpy as np
cimport heap

# determine datatypes for mcp
ctypedef np.float64_t FLOAT_T
ctypedef double FLOAT_C

cdef class MCP:
    cdef heap.FastUpdateBinaryHeap costs_heap
    cdef object costs_shape
    cdef int dim
    cdef object flat_costs
    cdef object flat_cumulative_costs
    cdef object traceback_offsets
    cdef object flat_edge_map
    cdef readonly object offsets
    cdef object flat_offsets
    cdef object offset_lengths
    cdef int dirty
    cdef int use_start_cost 
    # if use_start_cost is true, the cost of the starting element is added to
    # the cost of the path. Set to true by default in the base class...

    cdef FLOAT_C _travel_cost(self, FLOAT_C old_cost, FLOAT_C new_cost, FLOAT_C offset_length)
