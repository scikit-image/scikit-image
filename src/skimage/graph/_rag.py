from _skimage2.graph._rag import (
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

from skimage._doctest_adapters import adapt_doctests, returns_shim

adapt_doctests(globals())

rag_mean_color = returns_shim(rag_mean_color)
rag_boundary = returns_shim(rag_boundary)
