""" non local means filter

fast implementation see (fix ref)


"""
from __future__ import print_function, unicode_literals, absolute_import, division
import logging
logger = logging.getLogger(__name__)


import numpy as np

from gputools import OCLArray,OCLImage, OCLProgram, get_device

from ._abspath import abspath


def nlm2(data,sigma, size_filter = 2, size_search = 3):
    """for noise level of sigma_0, choose sigma = 1.5*sigma_0
    
    """

    prog = OCLProgram(abspath("kernels/nlm2.cl"),
                      build_options="-D FS=%i -D BS=%i"%(size_filter,size_search))

    data = data.astype(np.float32)
    img = OCLImage.from_array(data)

    distImg = OCLImage.empty_like(data)

    distImg = OCLImage.empty_like(data)
    tmpImg = OCLImage.empty_like(data)
    tmpImg2 = OCLImage.empty_like(data)

    accBuf = OCLArray.zeros(data.shape,np.float32)    
    weightBuf = OCLArray.zeros(data.shape,np.float32)

    for dx in range(size_search+1):
        for dy in range(-size_search,size_search+1):
                prog.run_kernel("dist",img.shape,None,
                                img,tmpImg,np.int32(dx),np.int32(dy))
                prog.run_kernel("convolve",img.shape,None,
                                tmpImg,tmpImg2,np.int32(1))
                prog.run_kernel("convolve",img.shape,None,
                                tmpImg2,distImg,np.int32(2))

                prog.run_kernel("computePlus",img.shape,None,
                                img,distImg,accBuf.data,weightBuf.data,
                               np.int32(img.shape[0]),np.int32(img.shape[1]),
                               np.int32(dx),np.int32(dy),np.float32(sigma))

                if dx!=0:
                #if any([dx,dy]):
                    prog.run_kernel("computeMinus",img.shape,None,
                                    img,distImg,accBuf.data,weightBuf.data,
                               np.int32(img.shape[0]),np.int32(img.shape[1]),
                               np.int32(dx),np.int32(dy),np.float32(sigma))

    accBuf /= weightBuf
    return accBuf.get()



if __name__ == '__main__':
    pass
