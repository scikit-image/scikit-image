# -*- coding: utf-8 -*-
# uft.py --- Unitary fourier transform

# Copyright (c) 2011, 2012, 2013  Fran��ois Orieux <orieux@iap.fr>

# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Commentary:

"""Function of unitary fourier transform and utilities

This module implement unitary fourier transform, that is ortho-normal
transform. They are specially usefull for convolution [1]: they
respect the parseval equality, the value of the null frequency is
equal to

.. math:: \frac{1}{\sqrt{n}} \sum_i x_i.

If the anfft module is present, his function are used. anfft wrap fftw
C library. Otherwise, numpy fft functions are used.

You must keep in mind that the transform are applied from the last
axes. this is a fftw convention for performance reason (c order
array). If you want more sofisticated use, you must use directly the
numpy.fft module.

References
----------
.. [1] B. R. Hunt "A matrix theory proof of the discrete convolution
       theorem", IEEE Trans. on Audio and Electroacoustics,
       vol. au-19, no. 4, pp. 285-288, dec. 1971

"""

# code:

import logging

import numpy as np
try:
    import anfft
    ANFFTMOD = True
except ImportError:
    logging.info("Installation of the anfft package improve preformance"
                 " by using fftw library.")
    ANFFTMOD = False

__author__ = "Fran��ois Orieux"
__copyright__ = "Copyright (C) 2011, 2012, 2013 F. Orieux <orieux@iap.fr>"
__credits__ = ["Fran��ois Orieux"]
__license__ = "mit"
__version__ = "0.1.0"
__maintainer__ = "Fran��ois Orieux"
__email__ = "orieux@iap.fr"
__status__ = "development"
__url__ = ""
__keywords__ = "fft"


def _circshift(inarray, shifts):
    """Shift array circularly.

    Circularly shifts the values in the array `a` by `s`
    elements. Return a copy.

    Parameters
    ----------
    a : ndarray
       The array to shift.

    s : tuple of int
       A tuple of integer scalars where the N-th element specifies the
       shift amount for the N-th dimension of array `a`. If an element
       is positive, the values of `a` are shifted down (or to the
       right). If it is negative, the values of `a` are shifted up (or
       to the left).

    Returns
    -------
    y : ndarray
       The shifted array (elements are copied)

    Examples
    --------
    >>> circshift(np.arange(10), 2)
    array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7])

    """
    # Initialize array of indices
    idx = []

    # Loop through each dimension of the input matrix to calculate
    # shifted indices
    for dim in range(inarray.ndim):
        length = inarray.shape[dim]
        try:
            shift = shifts[dim]
        except IndexError:
            shift = 0  # no shift if not specify

        # Lets start for fancy indexing. First we build the shifted
        # index for dim k. It will be broadcasted to other dim so
        # ndmin is specified
        index = np.mod(np.array(range(length),
                                ndmin=inarray.ndim) - shift,
                       length)
        # Shape adaptation
        shape = np.ones(inarray.ndim)
        shape[dim] = inarray.shape[dim]
        index = np.reshape(index, shape)

        idx.append(index.astype(int))

    # Perform the actual conversion by indexing into the input matrix
    return inarray[idx]


def ufftn(inarray, dim=None):
    """N-dim unitary Fourier transform

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    dim : int, optional
        The `dim` last axis along wich to compute the transform. All
        axes by default.

    Returns
    -------
    outarray : array-like (same shape than inarray)
    """
    if not dim:
        dim = inarray.ndim

    if ANFFTMOD:
        outarray = anfft.fftn(inarray, k=dim)
    else:
        outarray = np.fft.fftn(inarray, axes=range(-dim, 0))

    return outarray / np.sqrt(np.prod(inarray.shape[-dim:]))


def uifftn(inarray, dim=None):
    """N-dim unitary inverse Fourier transform

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    dim : int, optional
        The `dim` last axis along wich to compute the transform. All
        axes by default.

    Returns
    -------
    outarray : array-like (same shape than inarray)
    """
    if not dim:
        dim = inarray.ndim

    if ANFFTMOD:
        outarray = anfft.ifftn(inarray, k=dim)
    else:
        outarray = np.fft.ifftn(inarray, axes=range(-dim, 0))

    return outarray * np.sqrt(np.prod(inarray.shape[-dim:]))


def urfftn(inarray, dim=None):
    """N-dim real unitary Fourier transform

    This transform consider the Hermitian property of the transform on
    real input

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    dim : int, optional
        The `dim` last axis along wich to compute the transform. All
        axes by default.

    Returns
    -------
    outarray : array-like (the last dim as  N / 2 + 1 lenght)
    """
    if not dim:
        dim = inarray.ndim

    if ANFFTMOD:
        outarray = anfft.rfftn(inarray, k=dim)
    else:
        outarray = np.fft.rfftn(inarray, axes=range(-dim, 0))

    return outarray / np.sqrt(np.prod(inarray.shape[-dim:]))


def uirfftn(inarray, dim=None):
    """N-dim real unitary Fourier transform

    This transform consider the Hermitian property of the transform
    from complex to real real input.

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    dim : int, optional
        The `dim` last axis along wich to compute the transform. All
        axes by default.

    Returns
    -------
    outarray : array-like (the last dim as (N - 1) *2 lenght)
    """
    if not dim:
        dim = inarray.ndim

    if ANFFTMOD:
        outarray = anfft.irfftn(inarray, k=dim)
    else:
        outarray = np.fft.irfftn(inarray, axes=range(-dim, 0))

    return outarray * np.sqrt(np.prod(inarray.shape[-dim:-1]) *
                              (inarray.shape[-1] - 1) * 2)


def ufft2(inarray):
    """2-dim unitary Fourier transform

    Compute the Fourier transform on the last 2 axes.

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    Returns
    -------
    outarray : array-like (same shape than inarray)

    See Also
    --------
    uifft2, ufftn, urfftn
    """
    return ufftn(inarray, 2)


def uifft2(inarray):
    """2-dim inverse unitary Fourier transform

    Compute the inverse Fourier transform on the last 2 axes.

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    Returns
    -------
    outarray : array-like (same shape than inarray)

    See Also
    --------
    uifft2, uifftn, uirfftn
    """
    return uifftn(inarray, 2)


def urfft2(inarray):
    """2-dim real unitary Fourier transform

    Compute the real Fourier transform on the last 2 axes. This
    transform consider the Hermitian property of the transform from
    complex to real real input.

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    Returns
    -------
    outarray : array-like (the last dim as (N - 1) *2 lenght)

    See Also
    --------
    ufft2, ufftn, urfftn
    """
    return urfftn(inarray, 2)


def uirfft2(inarray):
    """2-dim real unitary Fourier transform

    Compute the real inverse Fourier transform on the last 2 axes.
    This transform consider the Hermitian property of the transform
    from complex to real real input.

    Parameters
    ----------
    inarray : ndarray
        The array to transform.

    Returns
    -------
    outarray : array-like (the last dim as (N - 1) *2 lenght)

    See Also
    --------
    urfft2, uifftn, uirfftn
    """
    return uirfftn(inarray, 2)


def image_quad_norm(inarray):
    """Return quadratic norm of images in Fourier space

    This function detect if the image suppose the hermitian property.

    Parameters
    ----------
    inarray : array-like
        The images are supposed to be in the last two axes

    Returns
    -------
    norm : float

    """
    # If there is an hermitian symmetry
    if inarray.shape[-1] != inarray.shape[-2]:
        return 2 * np.sum(np.sum(np.abs(inarray)**2, axis=-1), axis=-1) - \
            np.sum(np.abs(inarray[..., 0])**2, axis=-1)
    else:
        return np.sum(np.sum(np.abs(inarray)**2, axis=-1), axis=-1)


def crandn(shape):
    """white complex gaussian noise

    Generate directly the unitary Fourier transform of white gaussian
    noise noise field (with given shape) of zero mean and variance
    unity (ie N(0,1)).
    """
    return np.sqrt(0.5) * (np.random.standard_normal(shape) +
                           1j * np.random.standard_normal(shape))


def ir2tf(imp_resp, shape, dim=None, real=True):
    """Compute the transfer function of IR

    This function make the necessary correct zero-padding, zero
    convention, correct fft2 etc... to compute the transfer function
    of IR. To use with unitary Fourier transform for the signal (ufftn
    or equivalent).

    Parameters
    ----------
    imp_resp : ndarray
       The impulsionnal responses.

    shape : tuple of int
       A tuple of integer corresponding to the target shape of the
       tranfert function.

    dim : int, optional
        The `dim` last axis along wich to compute the transform. All
        axes by default.

    real : boolean (optionnal, default True)
       If True, imp_resp is supposed real and the hermissian property
       is used with rfftn Fourier transform.

    Returns
    -------
    y : complex ndarray
       The tranfert function of shape `shape`.

    See Also
    --------
    ufftn, uifftn, urfftn, uirfftn

    Notes
    -----
    The input array can be composed of multiple dimentionnal IR with
    an arbitraru number of IR. The individual IR must be accesed
    through first axes. The last `dim` axes of space definition. The
    `dim` parameter must be specified to compute the transform only
    along these last axes.
    """
    if not dim:
        dim = imp_resp.ndim

    # Zero padding and fill
    irpadded = np.zeros(shape)
    irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp
    # Circshift fo zero convention of the fft to avoid the phase
    # problem. Work with odd and even size.
    irpadded = _circshift(irpadded,
                          [-np.floor(s / 2)
                           if i >= imp_resp.ndim - dim
                           else 0
                           for i, s in enumerate(imp_resp.shape)])

    if real:
        if ANFFTMOD:
            return anfft.rfftn(irpadded, k=dim)
        else:
            return np.fft.rfftn(irpadded, axes=range(-dim, 0))
    else:
        if ANFFTMOD:
            return anfft.fftn(irpadded, k=dim)
        else:
            return np.fft.fftn(irpadded, axes=range(-dim, 0))


def laplacian(ndim, shape):
    """Return the transfert function of the laplacian

    Laplacian is the second order difference, on line and column.

    Parameters
    ----------
    ndim : int
        The dimension of the laplacian

    shape : tuple, shape
        The support on which to compute the transfert function

    Returns
    -------
    tf : array_like, complex
        The transfert function

    impr : array_like, real
        The laplacian
    """
    impr = np.zeros([3] * ndim)
    for dim in range(ndim):
        idx = tuple([slice(1, 2)] * dim +
                    [slice(None)] +
                    [slice(1, 2)] * (ndim - dim - 1))
        impr[idx] = np.array([-1.0,
                              0.0,
                              -1.0]).reshape([-1 if i == dim else 1
                                              for i in range(ndim)])
    impr[([slice(1, 2)] * ndim)] = 2.0 * ndim

    return ir2tf(impr, shape), impr
