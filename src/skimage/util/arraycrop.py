"""

The arraycrop module contains functions to crop values from the edges of an
n-dimensional array.

"""

from _skimage2.util.arraycrop import crop as crop  # noqa: F401

__all__ = ['crop']

from skimage._docutils import bind_namespace

bind_namespace(globals())
