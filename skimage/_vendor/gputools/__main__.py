from __future__ import print_function, unicode_literals, absolute_import, division

import gputools
import numpy as np
from itertools import product

def _convolve_rand(dshape,hshape):
    print("convolving test: dshape = %s, hshape  = %s"%(dshape,hshape))
    np.random.seed(1)
    d = np.random.uniform(-1,1,dshape).astype(np.float32)
    h = np.random.uniform(-1,1,hshape).astype(np.float32)
    out2 = gputools.convolve(d,h)

    
def test_convolve():
    for ndim in [1,2,3]:
        for N in range(10,200,40):
            for Nh in range(3,11,2):
                dshape = [N//ndim+3*n for n in range(ndim)]
                hshape = [Nh+3*n for n in range(ndim)]
                _convolve_rand(dshape,hshape)


def _compare_fft_np(d):
    res1 = np.fft.fftn(d)
    res2 = gputools.fft(d, fast_math=True)
    return res1, res2

def test_compare():
    for ndim in [1, 2, 3]:
        for dshape in product([32, 64, 128], repeat=ndim):
            d = np.random.uniform(-1, 1, dshape).astype(np.complex64)
            res1, res2 = _compare_fft_np(d)
            print("validating fft of size", d.shape)
            print("max relative error: %.4gf "%(np.amax(np.abs(res1-res2))/np.amax(np.abs(res1))))

if __name__ == '__main__':
    test_convolve()
    test_compare()
