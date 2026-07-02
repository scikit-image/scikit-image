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

# -- Shims for methods that internally construct _skimage2 class instances ----

# __add__ falls back to super().__thisclass__ (= _skimage2.ProjectiveTransform)
# when the two operand types differ.  The result is a plain _skimage2 instance
# that won't pass isinstance checks against the shim class.  Override __add__ on
# every transform proxy class so the returned instance is re-typed to the shim.
# (We cannot put the override only on ProjectiveTransform because the sibling
# proxy classes don't inherit from each other.)

_orig_add = ProjectiveTransform.__add__


def _shim_add(self, other):
    result = _orig_add(self, other)
    result_shim_cls = globals().get(type(result).__name__)
    if result_shim_cls is not None and type(result) is not result_shim_cls:
        result.__class__ = result_shim_cls
    return result


for _name in (
    'ProjectiveTransform',
    'AffineTransform',
    'SimilarityTransform',
    'EuclideanTransform',
):
    _proxy = globals().get(_name)
    if _proxy is not None:
        _proxy.__add__ = _shim_add


# isinstance() between sibling shim proxy classes doesn't work out of the box
# because each proxy inherits only from its own _skimage2 original (not from
# sibling proxies).  The _skimage2 classes use abc.ABC (→ ABCMeta), so the
# proxy classes inherit that metaclass.  Use ABCMeta.register() to make each
# shim proxy a virtual subclass of the shim proxy for the parent _skimage2
# class, so e.g. isinstance(shim_Essential_instance, shim_Fundamental) works.
_parent_map: dict[str, str] = {
    'EssentialMatrixTransform': 'FundamentalMatrixTransform',
    'FundamentalMatrixTransform': 'ProjectiveTransform',
    'ProjectiveTransform': 'ProjectiveTransform',  # self (no-op)
    'AffineTransform': 'ProjectiveTransform',
    'SimilarityTransform': 'ProjectiveTransform',
    'EuclideanTransform': 'ProjectiveTransform',
}
for _child_name, _parent_name in _parent_map.items():
    _child = globals().get(_child_name)
    _parent = globals().get(_parent_name)
    if _child is not None and _parent is not None and _child is not _parent:
        _parent.register(_child)

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
