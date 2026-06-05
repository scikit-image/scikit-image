"""
Scipy variant of Cython command

Cython, as applied to single pyx file.

Expects two arguments, infile and outfile.

Other options passed through to cython command line parser.

"""

from _skimage2._build_utils.cythoner import main as main  # noqa: F401

__all__ = ['main']
