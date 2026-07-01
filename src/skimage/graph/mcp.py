from _skimage2.graph.mcp import (
    MCP as MCP,
    MCP_Connect as MCP_Connect,
    MCP_Flexible as MCP_Flexible,
    MCP_Geometric as MCP_Geometric,
    route_through_array as route_through_array,
)  # noqa: F401

__all__ = [
    'MCP',
    'MCP_Connect',
    'MCP_Flexible',
    'MCP_Geometric',
    'route_through_array',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
