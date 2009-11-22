try:
    from spath import shortest_path
    from trace_path import trace_path
except ImportError:
    print """*** The shortest path extension has not been compiled.  Run

python setup.py build_ext -i

in the source directory to build in-place.  Please refer to INSTALL.txt
for further detail."""
