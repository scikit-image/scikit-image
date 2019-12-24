from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import logging
from collections import namedtuple

logger = logging.getLogger(__name__)

from gputools import OCLArray, get_device
from gputools.core.ocltypes import assert_bufs_type
import reikna.cluda as cluda
from reikna.fft import FFT

def _convert_axes_to_absolute(dshape, axes):
    """axes = (-2,-1) does not work in reikna, so we have to convetr that"""

    if axes is None:
        return None
    elif isinstance(axes, (tuple, list)):
        return tuple(np.arange(len(dshape))[list(axes)])
    else:
        raise NotImplementedError("axes %s is of unsupported type %s "%(str(axes), type(axes)))


class MockBuffer():
    """
    Used during creation of fft plan, as it expects a numpy array (which we dont want to create in the first place)
    """
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape


# def fft_plan_pyfft(shape, **kwargs):
#     """returns an opencl/pyfft plan of shape dshape
#
#     kwargs are the same as pyfft.cl.Plan
#     """
#     return Plan(shape, queue=get_device().queue, **kwargs)


def fft_plan(shape, dtype=np.complex64, axes=None, fast_math=True):
    """returns an reikna plan/FFT obj of shape dshape
    """
    # if not axes is None and any([a<0 for a in axes]):
    #     raise NotImplementedError("indices of axes have to be non negative, but are: %s"%str(axes))

    axes = _convert_axes_to_absolute(shape, axes)

    mock_buffer = MockBuffer(dtype, shape)

    fft_plan = FFT(mock_buffer, axes=axes).compile(cluda.ocl_api().Thread(get_device().queue),
                                                   fast_math=fast_math)

    return fft_plan


def fft(arr_obj, res_g=None,
        inplace=False,
        inverse=False,
        axes=None,
        plan=None,
        fast_math=True):
    """ (inverse) fourier trafo of 1-3D arrays

    creates a new plan or uses the given plan
    
    the transformed arr_obj should be either a

    - numpy array:

        returns the fft as numpy array (inplace is ignored)
    
    - OCLArray of type complex64:

        writes transform into res_g if given, to arr_obj if inplace
        or returns a new OCLArray with the transform otherwise
    
    """

    if plan is None:
        plan = fft_plan(arr_obj.shape, arr_obj.dtype,
                        axes=axes,
                        fast_math=fast_math)

    if isinstance(arr_obj, np.ndarray):
        return _ocl_fft_numpy(plan, arr_obj, inverse=inverse)
    elif isinstance(arr_obj, OCLArray):
        if not arr_obj.dtype.type is np.complex64:
            raise TypeError("OCLArray arr_obj has to be of complex64 type")

        if inplace:
            _ocl_fft_gpu_inplace(plan, arr_obj, inverse=inverse,
                                 )
        else:
            #FIXME
            raise NotImplementedError("currently only inplace fft is supported (FIXME)")

            return _ocl_fft_gpu(plan, arr_obj,
                                res_arr=res_g,
                                inverse=inverse,
                                )

    else:
        raise TypeError("array argument (1) has bad type: %s" % type(arr_obj))


# implementation ------------------------------------------------

def _ocl_fft_numpy(plan, arr, inverse=False, fast_math=True):
    if arr.dtype != np.complex64:
        logger.info("converting %s to complex64, might slow things down..." % arr.dtype)

    ocl_arr = OCLArray.from_array(arr.astype(np.complex64, copy=False))

    _ocl_fft_gpu_inplace(plan, ocl_arr, inverse=inverse)

    return ocl_arr.get()


def _ocl_fft_gpu_inplace(plan, ocl_arr, inverse=False):
    assert_bufs_type(np.complex64, ocl_arr)
    plan(ocl_arr, ocl_arr, inverse=inverse)


def _ocl_fft_gpu(plan, ocl_arr, res_arr=None, inverse=False):
    assert_bufs_type(np.complex64, ocl_arr)

    if res_arr is None:
        res_arr = OCLArray.empty_like(ocl_arr)
    plan(ocl_arr, res_arr, inverse=inverse)

    return res_arr


if __name__ == '__main__':
    d = np.random.uniform(0, 1, (64,) * 2).astype(np.complex64)

    b = OCLArray.from_array(d)

    plan = fft_plan(d.shape)

    d2 = fft(d, plan=plan)

    fft(b, inplace=True, plan=plan)
