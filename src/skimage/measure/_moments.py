from _skimage2.measure._moments import (
    centroid as centroid,
    inertia_tensor as inertia_tensor,
    inertia_tensor_eigvals as inertia_tensor_eigvals,
    moments as moments,
    moments_central as moments_central,
    moments_coords as moments_coords,
    moments_coords_central as moments_coords_central,
    moments_hu as moments_hu,
    moments_normalized as moments_normalized,
)  # noqa: F401

__all__ = [
    'centroid',
    'inertia_tensor',
    'inertia_tensor_eigvals',
    'moments',
    'moments_central',
    'moments_coords',
    'moments_coords_central',
    'moments_hu',
    'moments_normalized',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
