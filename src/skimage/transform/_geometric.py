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
    _affine_matrix_from_vector,
    _append_homogeneous_dim,
    _apply_homogeneous,
    _calc_center_normalize,
    _center_and_normalize_points,
    _euler_rotation_matrix,
)


from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())

# Fix __add__ results by explicit assignment of generic_projective_class.
# generic_projective_class gives the output class of added transforms, when the
# transform types added are not the same.
for _c in (
    ProjectiveTransform,
    AffineTransform,
    EuclideanTransform,
    SimilarityTransform,
):
    _c.generic_projective_class = ProjectiveTransform

# Use the fact that geometric classes inherit from ABC, and therefore implement
# 'register', to fix inheritance checks after subclassing in adapt_doctests.
ProjectiveTransform.register(AffineTransform)
ProjectiveTransform.register(EuclideanTransform)
EuclideanTransform.register(SimilarityTransform)
FundamentalMatrixTransform.register(EssentialMatrixTransform)

# Refill transforms from our own (shimmed) transform definitions.
TRANSFORMS = {k: globals()[v.__name__] for k, v in _SKI2_TRANSFORMS.items()}

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
