"""
Function of unitary fourier transform (uft) and utilities

This module implements the unitary fourier transform, also known as
the ortho-normal transform. It is especially useful for convolution
[1], as it respects the Parseval equality. The value of the null
frequency is equal to

.. math::  \\frac{1}{\\sqrt{n}} \\sum_i x_i

so the Fourier transform has the same energy as the original image
(see ``image_quad_norm`` function). The transform is applied from the
last axis for performance (assuming a C-order array input).

References
----------
.. [1] B. R. Hunt "A matrix theory proof of the discrete convolution
       theorem", IEEE Trans. on Audio and Electroacoustics,
       vol. au-19, no. 4, pp. 285-288, dec. 1971


"""

from _skimage2.restoration.uft import (
    image_quad_norm as image_quad_norm,
    ir2tf as ir2tf,
    laplacian as laplacian,
    ufft2 as ufft2,
    ufftn as ufftn,
    uifft2 as uifft2,
    uifftn as uifftn,
    uirfft2 as uirfft2,
    uirfftn as uirfftn,
    urfft2 as urfft2,
    urfftn as urfftn,
)  # noqa: F401

__all__ = [
    'image_quad_norm',
    'ir2tf',
    'laplacian',
    'ufft2',
    'ufftn',
    'uifft2',
    'uifftn',
    'uirfft2',
    'uirfftn',
    'urfft2',
    'urfftn',
]

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())
