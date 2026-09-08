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

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
