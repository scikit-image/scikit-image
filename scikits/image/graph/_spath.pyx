import _mcp
cimport _mcp

cdef extern from "math.h":
    double fabs(double f)
 
cdef class MCP_Diff(_mcp.MCP):
    """MCP_Diff(costs, offsets=None, fully_connected=True)
    
    Find minimum-difference paths through an n-d costs array.
    
    See the documentation for MCP for full details. This class differs from 
    MCP in that the cost of a path is not simply the sum of the costs along
    that path.
    
    This class instead assumes that the cost of moving from one point to
    another is the absolute value of the difference in the costs between the
    two points
    """
    def __init__(self, costs, offsets=None, fully_connected=True):
        """__init__(costs, offsets=None, fully_connected=True)
        
        See class documentation.
        """
        _mcp.MCP.__init__(self, costs, offsets, fully_connected)
        self.use_start_cost = 0
    
    cdef _mcp.FLOAT_C _travel_cost(self, _mcp.FLOAT_C old_cost, _mcp.FLOAT_C new_cost, _mcp.FLOAT_C offset_length):
        return  fabs(old_cost - new_cost)
