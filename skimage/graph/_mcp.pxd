""" This is the definition file for mcp.pyx.
It contains the definitions of the mcp class, such that
other cython modules can "cimport mcp" and subclass it.
"""

from . cimport heap
cimport numpy as cnp

ctypedef heap.BOOL_T BOOL_T
ctypedef unsigned char DIM_T
ctypedef cnp.float64_t FLOAT_T
ctypedef cnp.intp_t INDEX_T
ctypedef cnp.int8_t EDGE_T
ctypedef cnp.int8_t OFFSET_T
ctypedef cnp.int16_t OFFSETS_INDEX_T


cdef class MCP:
    cdef heap.FastUpdateBinaryHeap costs_heap
    cdef object costs_shape
    cdef object _starts
    cdef object _ends
    cdef DIM_T dim
    cdef BOOL_T dirty
    cdef BOOL_T use_start_cost
    # if use_start_cost is true, the cost of the starting element is added to
    # the cost of the path. Set to true by default in the base class...

    # Arrays used during front propagation
    cdef FLOAT_T [:] flat_costs
    cdef FLOAT_T [:] flat_cumulative_costs
    cdef OFFSETS_INDEX_T [:] traceback_offsets
    cdef EDGE_T [:,:] flat_pos_edge_map
    cdef EDGE_T [:,:] flat_neg_edge_map
    # offsets is part of public API. Used to interpret traceback result of find_costs()
    cdef public OFFSET_T [:,:] offsets
    cdef INDEX_T [:] flat_offsets
    cdef FLOAT_T [:] offset_lengths

    # Methods
    cpdef int goal_reached(self, INDEX_T index, FLOAT_T cumcost)
    cdef FLOAT_T _travel_cost(self, FLOAT_T old_cost, FLOAT_T new_cost, FLOAT_T offset_length)
    cdef void _examine_neighbor(self, INDEX_T index, INDEX_T new_index, FLOAT_T offset_length)
    cdef void _update_node(self, INDEX_T index, INDEX_T new_index, FLOAT_T offset_length)

