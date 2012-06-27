""" This is the definition file for mcp.pyx.
It contains the definitions of the mcp class, such that
other cython modules can "cimport mcp" and subclass it.
"""

cimport heap
cimport numpy as np

ctypedef heap.BOOL_T BOOL_T
ctypedef unsigned char DIM_T 
ctypedef np.float64_t FLOAT_T

cdef class MCP:
    cdef heap.FastUpdateBinaryHeap costs_heap
    cdef object costs_shape
    cdef DIM_T dim
    cdef object flat_costs
    cdef object flat_cumulative_costs
    cdef object traceback_offsets
    cdef object flat_pos_edge_map
    cdef object flat_neg_edge_map
    cdef readonly object offsets
    cdef object flat_offsets
    cdef object offset_lengths
    cdef BOOL_T dirty
    cdef BOOL_T use_start_cost 
    # if use_start_cost is true, the cost of the starting element is added to
    # the cost of the path. Set to true by default in the base class...

    cdef FLOAT_T _travel_cost(self, FLOAT_T old_cost, FLOAT_T new_cost, FLOAT_T offset_length)
