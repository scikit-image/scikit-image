from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np


from gputools import OCLArray, get_device
from gputools.fft.oclfft import fft, fft_plan
from gputools.core.oclalgos import OCLElementwiseKernel
from gputools.core.ocltypes import assert_bufs_type

_complex_multiply_kernel = OCLElementwiseKernel(
    """cfloat_t *a, cfloat_t * b""",
    """a[i] = cfloat_mul(a[i],b[i])""", "mult")


def fft_convolve(data, h, res_g = None,
                 plan = None, inplace = False,
                 kernel_is_fft = False,
                 kernel_is_fftshifted = False):
    
    """ convolves data with kernel h via FFTs

    
    data should be either a numpy array or a OCLArray (see doc for fft)
    both data and h should be same shape

    if data/h are OCLArrays, then:
        - type should be complex64
        - shape should be equal and power of two
        - h is assumed to be already fftshifted
         (otherwise set kernel_is_fftshifted  to true)
    
    """

    
    if isinstance(data,np.ndarray):
        return _fft_convolve_numpy(data, h,
                                   plan = plan,
                                   kernel_is_fft = kernel_is_fft,
                                   kernel_is_fftshifted = kernel_is_fftshifted)
    elif isinstance(data,OCLArray):
        return _fft_convolve_gpu(data,h, res_g = res_g,
                                 plan = plan, inplace = inplace,
                                 kernel_is_fft = kernel_is_fft)
    else:
        raise TypeError("array argument (1) has bad type: %s"%type(data))



def _fft_convolve_numpy(data, h, plan = None,
                        kernel_is_fft = False,
                        kernel_is_fftshifted = False):
    """ convolving via opencl fft for numpy arrays

    data and h must have the same size
    """

    if data.shape != h.shape:
        raise ValueError("data and kernel must have same size! %s vs %s "%(str(data.shape),str(h.shape)))

    
    data_g = OCLArray.from_array(data.astype(np.complex64))

    if not kernel_is_fftshifted:
        h = np.fft.fftshift(h)

    
    h_g = OCLArray.from_array(h.astype(np.complex64))
    res_g = OCLArray.empty_like(data_g)
    
    _fft_convolve_gpu(data_g,h_g,res_g = res_g,
                      plan = plan,
                      kernel_is_fft = kernel_is_fft)

    res =  abs(res_g.get())

    del data_g
    del h_g
    del res_g
    
    return res


def _fft_convolve_gpu(data_g, h_g, res_g = None,
                      plan = None, inplace = False,
                      kernel_is_fft = False):
    """ fft convolve for gpu buffer
    """




    assert_bufs_type(np.complex64,data_g,h_g)

    if data_g.shape != h_g.shape:
        raise ValueError("data and kernel must have same size! %s vs %s "%(str(data_g.shape),str(h_g.shape)))


    if plan is None:
        plan = fft_plan(data_g.shape)

    if inplace:
        res_g = data_g
    else:
        if res_g is None:
            res_g = OCLArray.empty(data_g.shape,data_g.dtype)
            
        res_g.copy_buffer(data_g)
        
    if not kernel_is_fft:
        kern_g = OCLArray.empty(h_g.shape,h_g.dtype)
        kern_g.copy_buffer(h_g)
        fft(kern_g,inplace=True, plan = plan)
    else:
        kern_g = h_g


    fft(res_g,inplace=True, plan = plan)

    #multiply in fourier domain
    _complex_multiply_kernel(res_g,kern_g)

    fft(res_g,inplace = True, inverse = True, plan = plan)

    return res_g

    

if __name__ == '__main__':

    N = 512
    
    d = np.zeros((N,)*2)
    d[N//2,N/2] = 1.

    h = np.zeros((N,)*2)
    h[N//3:2*N//3,N//3:2*N//3] = 1.

    h = np.fft.fftshift(h)
        
    d_g = OCLArray.from_array(d.astype(np.complex64))


    h_g = OCLArray.from_array(h.astype(np.complex64))
    hf_g = OCLArray.from_array(np.fft.fft2(h.astype(np.complex64)))

    # out = fft_convolve(d_g,h_g, inplace = False, kernel_is_fft = True)

    out_g = fft_convolve(d_g,h_g, inplace = False)

    out = fft_convolve(d,h, inplace = False)

    
    print(np.sum(abs(out_g.get())),N**2/9)
