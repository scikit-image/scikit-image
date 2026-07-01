"""

This is an implementation of the marching cubes algorithm proposed in:

Efficient implementation of Marching Cubes' cases with topological guarantees.
Thomas Lewiner, Helio Lopes, Antonio Wilson Vieira and Geovan Tavares.
Journal of Graphics Tools 8(2): pp. 1-15 (december 2003)

This algorithm has the advantage that it provides topologically correct
results, and the algorithms implementation is relatively simple. Most
of the magic is in the lookup tables, which are provided as open source.

Originally implemented in C++ by Thomas Lewiner in 2002, ported to Cython
by Almar Klein in 2012. Adapted for scikit-image in 2016.


"""

from _skimage2.measure._marching_cubes_lewiner_cy import (
    Cell as Cell,
    Lut as Lut,
    LutProvider as LutProvider,
    marching_cubes as marching_cubes,
    remove_degenerate_faces as remove_degenerate_faces,
)  # noqa: F401

__all__ = [
    'Cell',
    'Lut',
    'LutProvider',
    'marching_cubes',
    'remove_degenerate_faces',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
