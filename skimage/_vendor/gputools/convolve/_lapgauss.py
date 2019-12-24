"""laplacian of gaussians for 2d and 3d data"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import imgtools

def lapGauss(data,sigma):
    if data.ndim==2:
        return _lapGauss2(data,sigma)
    elif data.ndim==3:
        return _lapGauss3(data,sigma)
    else:
        raise ValueError("data dim should be either 2 or 3 but is %s"%(data.ndim))


def _lapGauss2(y,sigma):
    N = int(4*sigma+1)
    x = N*np.linspace(-1,1.,N)
    h = np.exp(-x**2/2./sigma**2)
    h*= 1./np.sum(h)
    hx = (x**2/sigma**2-1.)/sigma**2*h
    outx = imgtools.convolve_sep2(y,hx,h)
    outy = imgtools.convolve_sep2(y,h,hx)
    return outx+outy


def _lapGauss3(y,sigma):
    N = int(4*sigma+1)
    x = N*np.linspace(-1,1.,N)
    h = np.exp(-x**2/2./sigma**2)
    h*= 1./np.sum(h)
    hx = (x**2/sigma**2-1.)/sigma**2*h
    outx = imgtools.convolve_sep3(y,hx,h,h)
    outy = imgtools.convolve_sep3(y,h,hx,h)
    outz = imgtools.convolve_sep3(y,h,h,hx)
    return outx+outy+outz



if __name__ == '__main__':
    pass
    
    
