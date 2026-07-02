from _skimage2.measure._colocalization import (
    pearson_corr_coeff as pearson_corr_coeff,
    manders_coloc_coeff as manders_coloc_coeff,
    manders_overlap_coeff as manders_overlap_coeff,
    intersection_coeff as intersection_coeff,
)  # noqa: F401

__all__ = [
    'pearson_corr_coeff',
    'manders_coloc_coeff',
    'manders_overlap_coeff',
    'intersection_coeff',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
