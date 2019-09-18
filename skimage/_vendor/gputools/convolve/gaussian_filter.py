from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
from gputools import OCLArray
from .convolve import convolve
from .convolve_sep import convolve_sep2, convolve_sep3


def gaussian_filter(data, sigma=4., truncate = 4., normalize=True, res_g=None):
    """
    blurs data with a gaussian kernel of given sigmas
    
    Parameters
    ----------
    data: ndarray
        2 or 3 dimensional array 
    sigma: scalar or tuple
        the sigma of the gaussian  
    truncate: float 
        truncate the kernel after truncate*sigma  
    normalize: bool
        uses a normalized kernel is true
    res_g: OCLArray
        used to store result if given  

    Returns
    -------
        blurred array 
    """

    if not len(data.shape) in [1, 2, 3]:
        raise ValueError("dim = %s not supported" % (len(data.shape)))

    if np.isscalar(sigma):
        sigma = [sigma] * data.ndim

    if any(tuple(s <= 0 for s in sigma)):
        raise ValueError("sigma = %s : all sigmas have to be positive!" % str(sigma))

    if isinstance(data, OCLArray):
        return _gaussian_buf(data, sigma, res_g, normalize=normalize,truncate = truncate)
    elif isinstance(data, np.ndarray):
        return _gaussian_np(data, sigma,  normalize=normalize,truncate = truncate)

    else:
        raise TypeError("unknown type (%s)" % (type(data)))


def _gaussian_buf(d_g, sigma=(4., 4.),  res_g=None, normalize=True,truncate = 4.0):
    radius = tuple(int(truncate*s +0.5) for s in sigma)

    ns = tuple(np.arange(-r,r+1) for r in radius)


    hs = tuple(
        np.exp(-.5 / s ** 2 * n**2) for s, n in zip(reversed(sigma), reversed(ns)))

    if normalize:
        hs = tuple(1. * h / np.sum(h) for h in hs)

    h_gs = tuple(OCLArray.from_array(h.astype(np.float32)) for h in hs)


    if len(d_g.shape) == 1:
        return convolve(d_g, *h_gs, res_g=res_g)
    elif len(d_g.shape) == 2:
        return convolve_sep2(d_g, *h_gs, res_g=res_g)
    elif len(d_g.shape) == 3:
        return convolve_sep3(d_g, *h_gs, res_g=res_g)

    else:
        pass


def _gaussian_np(data, sigma,  normalize=True, truncate = 4.0):
    d_g = OCLArray.from_array(data.astype(np.float32, copy=False))

    return _gaussian_buf(d_g, sigma, truncate = truncate, normalize=normalize).get()


if __name__ == "__main__":
    pass
