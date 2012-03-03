try:
    from .spath import shortest_path
    from .mcp import MCP, MCP_Geometric, route_through_array
except ImportError:
    print """*** The cython extensions have not been compiled.  Run

python setup.py build_ext -i

in the source directory to build in-place.  Please refer to INSTALL.txt
for further detail."""
