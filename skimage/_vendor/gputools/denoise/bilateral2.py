"""
bilateral filter

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import logging
logger = logging.getLogger(__name__)


import numpy as np

from gputools import OCLArray,OCLImage, OCLProgram, get_device

from ._abspath import abspath


def bilateral2(data, fSize, sigma_p, sigma_x = 10.):
    """bilateral filter """
    
    dtype = data.dtype.type
    dtypes_kernels = {np.float32:"bilat2_float",
                        np.uint16:"bilat2_short"}

    if not dtype in dtypes_kernels:
        logger.info("data type %s not supported yet (%s), casting to float:"%(dtype,list(dtypes_kernels.keys())))
        data = data.astype(np.float32)
        dtype = data.dtype.type


    img = OCLImage.from_array(data)
    res = OCLArray.empty_like(data)

    
    prog = OCLProgram(abspath("kernels/bilateral2.cl"))


    prog.run_kernel(dtypes_kernels[dtype],
                    img.shape,None,
                    img,res.data,
                    np.int32(img.shape[0]),np.int32(img.shape[1]),
                    np.int32(fSize),np.float32(sigma_x),np.float32(sigma_p))


    return res.get()



if __name__ == '__main__':
    from scipy.misc import lena
    
    d = lena()

    d = np.random.poisson(d,d.shape)

    res = bilateral2(d,3,100.)
