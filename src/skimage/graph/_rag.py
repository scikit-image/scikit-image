import functools

from _skimage2.graph._rag import (
    RAG as RAG,
    min_weight as min_weight,
    rag_boundary as rag_boundary,
    rag_mean_color as rag_mean_color,
    show_rag as show_rag,
)  # noqa: F401

from _skimage2.graph._rag import _rag_boundary, _rag_mean_color

__all__ = [
    'RAG',
    'min_weight',
    'rag_boundary',
    'rag_mean_color',
    'show_rag',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())


@functools.wraps(rag_mean_color)
def rag_mean_color(image, labels, connectivity=2, mode='distance', sigma=255.0):
    return _rag_mean_color(RAG, image, labels, connectivity, mode, sigma)


@functools.wraps(rag_boundary)
def rag_boundary(labels, edge_map, connectivity=2):
    return _rag_boundary(RAG, labels, edge_map, connectivity)
