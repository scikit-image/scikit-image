"""

Cython implementation of Dijkstra's minimum cost path algorithm,
for use with data on a n-dimensional lattice.

"""

from _skimage2.graph._mcp import (
    DEPRECATED as DEPRECATED,
    EDGE_D as EDGE_D,
    FLOAT_D as FLOAT_D,
    INDEX_D as INDEX_D,
    MCP as MCP,
    MCP_Connect as MCP_Connect,
    MCP_Flexible as MCP_Flexible,
    MCP_Geometric as MCP_Geometric,
    OFFSETS_INDEX_D as OFFSETS_INDEX_D,
    OFFSET_D as OFFSET_D,
    deprecate_parameter as deprecate_parameter,
    heap as heap,
    make_offsets as make_offsets,
    warn as warn,
)  # noqa: F401

__all__ = [
    'DEPRECATED',
    'EDGE_D',
    'FLOAT_D',
    'INDEX_D',
    'MCP',
    'MCP_Connect',
    'MCP_Flexible',
    'MCP_Geometric',
    'OFFSETS_INDEX_D',
    'OFFSET_D',
    'deprecate_parameter',
    'heap',
    'make_offsets',
    'warn',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
