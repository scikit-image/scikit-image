from ._mcp import MCP, MCP_Geometric, make_offsets


def route_through_array(array, start, end, fully_connected=True, geometric=True):
    """Simple example of how to use the MCP and MCP_Geometric classes.

    See the MCP and MCP_Geometric class documentation for explanation of the
    path-finding algorithm.

    Parameters
    ----------
    array : ndarray
        Array of costs.
    start : iterable
        n-d index into `array` defining the starting point
    end : iterable
        n-d index into `array` defining the end point
    fully_connected : bool (optional)
        If True, diagonal moves are permitted, if False, only axial moves.
    geometric : bool (optional)
        If True, the MCP_Geometric class is used to calculate costs, if False,
        the MCP base class is used. See the class documentation for
        an explanation of the differences between MCP and MCP_Geometric.

    Returns
    -------
    path : list
        List of n-d index tuples defining the path from `start` to `end`.
    cost : float
        Cost of the path.
    """
    start, end = tuple(start), tuple(end)
    if geometric:
        mcp_class = MCP_Geometric
    else:
        mcp_class = MCP
    m = mcp_class(array, fully_connected=fully_connected)
    costs, traceback_array = m.find_costs([start], [end])
    return m.traceback(end), costs[end]
