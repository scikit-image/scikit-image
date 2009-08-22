"""
:author: Stefan van der Walt, 2008
:license: modified BSD
"""

__all__ = ['LPIFilter2D']
__docformat__ = 'restructuredtext en'

import numpy as np
from scipy.fftpack import fftshift, ifftshift

eps = np.finfo(float).eps

class LPIFilter2D(object):
    """Linear Position-Invariant Filter (2-dimensional)

    """
    def __init__(self,impulse_response,**filter_params):
        """
        *Parameters*:
            impulse_response : callable f(r,c,**filter_params)
                Function that yields the impulse response.  `r` and
                `c` are 1-dimensional vectors that represent row and
                column positions, in other words coordinates are
                (r[0],c[0]),(r[0],c[1]) etc.  `**filter_params` are
                passed through.

                In other words, example would be called like this:

                r = [0,0,0,1,1,1,2,2,2]
                c = [0,1,2,0,1,2,0,1,2]
                impulse_response(r,c,**filter_params)

        *Example*:

           Gaussian filter:

           >>> def filt_func(r,c):
                   return np.exp(-np.hypot(r,c)/1)

           >>> filter = LPIFilter2D(filt_func)


        """
        self.impulse_response = impulse_response
        self.filter_params = filter_params
        self._cache = None

    def _pad(self,data,shape):
        """Pad the data to the given shape with zeros.

        *Parameters*:
            data : 2-d ndarray
                Input data
            shape : (2,) tuple

        """
        out = np.zeros(shape)
        out[[slice(0,n) for n in data.shape]] = data
        return out

    def _prepare(self,data):
        """Calculate filter and data FFT in preparation for filtering.

        """
        dshape = np.array(data.shape)
        dshape += (dshape %2 == 0) # all filter dimensions must be uneven
        oshape = np.array(data.shape)*2-1

        if self._cache is None or np.any(self._cache.shape != oshape):
            coords = np.mgrid[[slice(0,float(n)) for n in dshape]]
            # this steps over two sets of coordinates,
            # not over the coordinates individually
            for k,coord in enumerate(coords):
                coord -= (dshape[k]-1)/2.
            coords = coords.reshape(2,-1).T # coordinate pairs (r,c)

            f = self.impulse_response(coords[:,0],coords[:,1],
                                      **self.filter_params).reshape(dshape)

            f = self._pad(f,oshape)
            F = np.dual.fftn(f)
            self._cache = F
        else:
            F = self._cache

        data = self._pad(data,oshape)
        G = np.dual.fftn(data)

        return F,G

    def _min_limit(self,x,val=eps):
        mask = np.abs(x) < eps
        x[mask] = np.sign(x[mask])*eps

    def _centre(self,x,oshape):
        """Return an array of oshape from the centre of x.

        """
        start = (np.array(x.shape) - np.array(oshape))/2.+1
        out = x[[slice(s,s+n) for s,n in zip(start,oshape)]]
        return out

    def __call__(self,data):
        """Apply the filter to the given data.

        *Parameters*:
            data : (M,N) ndarray

        """
        F,G = self._prepare(data)
        out = np.dual.ifftn(F*G)
        out = np.abs(self._centre(out,data.shape))
        return out

    def inverse(self,data,max_gain=2):
        """Apply the filter in reverse to the given data.

        *Parameters*:
            data : (M,N) ndarray
                Input data.
            max_gain : float
                Limit the filter gain.  Often, the filter contains
                zeros, which would cause the inverse filter to have
                infinite gain.  High gain causes amplification of
                artefacts, so a conservative limit is recommended.

        """
        F,G = self._prepare(data)
        self._min_limit(F)

        F = 1/F
        mask = np.abs(F) > max_gain
        F[mask] = np.sign(F[mask])*max_gain

        return self._centre(np.abs(ifftshift(np.dual.ifftn(G*F))),data.shape)

    def wiener(self,data,K=0.25):
        """Minimum Mean Square Error (Wiener) inverse filter.

        *Parameters*:
            data : (M,N) ndarray
                Input data.
            K : float or (M,N) ndarray
                Ratio between power spectrum of noise and undegraded
                image.

        """
        F,G = self._prepare(data)
        self._min_limit(F)

        H_mag_sqr = np.abs(F)**2
        F = 1/F * H_mag_sqr / (H_mag_sqr + K)

        return self._centre(np.abs(ifftshift(np.dual.ifftn(G*F))),data.shape)

    def constrained_least_squares(self,data,lam):
        pass
