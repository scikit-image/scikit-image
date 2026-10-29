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

# Use the fact that geometric classes inherit from ABC, and therefore implement
# 'register', to fix inheritance checks after subclassing in adapt_doctests.
ProjectiveTransform.register(AffineTransform)
ProjectiveTransform.register(EuclideanTransform)
EuclideanTransform.register(SimilarityTransform)
FundamentalMatrixTransform.register(EssentialMatrixTransform)


# ProjectiveTransform and child instance __add__ results will fail without
# intervention, as they depend on the definition of the ProjectiveTransform
# class as the default output, and this is the `_skimage2` version in the
# `_skimage2` tree.  Redefine __add__ (with exact same code as `_skimage2` at
# time of writing), so it picks up our ProjectiveTransform.
def _ptf__add__(self, other):
    """Combine this transformation with another."""
    # Is the other a projective instance?
    if not isinstance(other, ProjectiveTransform):
        raise TypeError("Cannot combine transformations of non-projective types.")
    # Combination of the same types result in a transformation of this
    # type again, otherwise the generic projective transform.
    tform_class = type(self) if type(self) == type(other) else ProjectiveTransform
    return tform_class(other.params @ self.params)


# We need to define __add__ for each projective transform, as the shims inherit
# from their original classes, so e.g.shim AffineTransform no longer inherits
# directly from shim ProjectiveTransform, and therefore, doesn't pick up
# `__add__` defined on ProjectiveTransform.
for _c in (
    ProjectiveTransform,
    AffineTransform,
    EuclideanTransform,
    SimilarityTransform,
):
    _c.__add__ = _ptf__add__


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
