from _skimage2.transform._geometric import (
    AffineTransform as AffineTransform,
    EssentialMatrixTransform as EssentialMatrixTransform,
    EuclideanTransform as EuclideanTransform,
    FundamentalMatrixTransform as FundamentalMatrixTransform,
    PiecewiseAffineTransform as PiecewiseAffineTransform,
    PolynomialTransform as PolynomialTransform,
    ProjectiveTransform as ProjectiveTransform,
    SimilarityTransform as SimilarityTransform,
    TRANSFORMS as _SKI2_TRANSFORMS,
    estimate_transform as estimate_transform,
    matrix_transform as matrix_transform,
)  # noqa: F401

from _skimage2.transform._geometric import (  # noqa: F401
    _GeometricTransform,
    _HMatrixTransform,
    _affine_matrix_from_vector,
    _append_homogeneous_dim,
    _apply_homogeneous,
    _calc_center_normalize,
    _center_and_normalize_points,
    _euler_rotation_matrix,
)


from skimage._doctest_adapters import adapt_doctests, as_shim, returns_shim

adapt_doctests(globals())


# The `_skimage2` `__add__` returns the `_skimage2` ProjectiveTransform when
# the two transform types differ.
_ski2_ptf__add__ = ProjectiveTransform.__add__


def _ptf__add__(self, other):
    """Combine this transformation with another."""
    return as_shim(_ski2_ptf__add__(self, other))


# The other projective transforms inherit `__add__` from the shim
# ProjectiveTransform.
ProjectiveTransform.__add__ = _ptf__add__


# Refill transforms from our own (shimmed) transform definitions.
TRANSFORMS = {k: globals()[v.__name__] for k, v in _SKI2_TRANSFORMS.items()}

PiecewiseAffineTransform._affine_class = AffineTransform

estimate_transform = returns_shim(estimate_transform)


__all__ = [
    'AffineTransform',
    'EssentialMatrixTransform',
    'EuclideanTransform',
    'FundamentalMatrixTransform',
    'PiecewiseAffineTransform',
    'PolynomialTransform',
    'ProjectiveTransform',
    'SimilarityTransform',
    'TRANSFORMS',
    'estimate_transform',
    'matrix_transform',
]
