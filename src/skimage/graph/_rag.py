import functools

# `_Ski2RAG` keeps the implementation class for the identity check in `_as_v1_rag`.
from _skimage2.graph._rag import (
    RAG as _Ski2RAG,
    RAG as RAG,
    min_weight as min_weight,
    rag_boundary as rag_boundary,
    rag_mean_color as rag_mean_color,
    show_rag as show_rag,
)  # noqa: F401

__all__ = [
    'RAG',
    'min_weight',
    'rag_boundary',
    'rag_mean_color',
    'show_rag',
]

from skimage._doctest_adapters import adapt_doctests

_ski2_rag_boundary = rag_boundary
_ski2_rag_mean_color = rag_mean_color

adapt_doctests(
    globals(),
    skip_names=('_Ski2RAG', '_ski2_rag_boundary', '_ski2_rag_mean_color'),
)


def _as_v1_rag(rag):
    """Return `rag` as an instance of the `skimage` RAG subclass."""
    # The implementation constructs the `_skimage2` RAG class; the shim
    # subclass adds no attributes, so we can swap the class in place.
    if type(rag) is _Ski2RAG:
        rag.__class__ = RAG
    return rag


@functools.wraps(rag_mean_color)
def rag_mean_color(*args, **kwargs):
    return _as_v1_rag(_ski2_rag_mean_color(*args, **kwargs))


@functools.wraps(rag_boundary)
def rag_boundary(*args, **kwargs):
    return _as_v1_rag(_ski2_rag_boundary(*args, **kwargs))


rag_mean_color.__module__ = __name__
rag_boundary.__module__ = __name__
