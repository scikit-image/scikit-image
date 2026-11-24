from _skimage2.graph._spath import MCP_Diff as MCP_Diff  # noqa: F401

__all__ = ['MCP_Diff']

# Make the shim class for the base class first.
from . import _mcp  # noqa: F401
from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
