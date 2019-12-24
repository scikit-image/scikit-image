""" non local means filter

fast implementation see (fix ref)


"""
from __future__ import print_function, unicode_literals, absolute_import, division
import logging
logger = logging.getLogger(__name__)


import numpy as np

from gputools import OCLArray,OCLImage, OCLProgram, get_device

from ._abspath import abspath


def nlm3(data,sigma, size_filter = 2, size_search = 3):
    """
    Fast version of Non local mean denoising of 3 dimensional data
    see [1]_

    Parameters
    ----------
    data: 3d ndarray
        the input volume
    sigma: float
        denoising strength
    size_filter: int
        the half size of the image patches (i.e. width is 2*size_filter+1 along every dimension)
    size_search: int
        the half size of the search window (i.e. width is 2*size_search+1 along every dimension)

    Returns
    -------
    ndarray
        the denoised volume

    Examples
    --------

    >>> d = np.random.uniform(0,1,(100,)*3)
    >>> d[40:60,40:60,40:60] += 5
    >>> res = nlm3(d,1.,3,4)

    References
    ----------

    .. [1] Buades, Antoni, Bartomeu Coll, and J-M. Morel. "A non-local algorithm for image denoising." CVPR 2005.
    """

    prog = OCLProgram(abspath("kernels/nlm3.cl"),
                      build_options="-D FS=%i -D BS=%i"%(size_filter,size_search))


    data = data.astype(np.float32, copy = False)
    img = OCLImage.from_array(data)

    distImg = OCLImage.empty_like(data)
    tmpImg = OCLImage.empty_like(data)
    tmpImg2 = OCLImage.empty_like(data)

    accBuf = OCLArray.zeros(data.shape,np.float32)    
    weightBuf = OCLArray.zeros(data.shape,np.float32)

    for dx in range(size_search+1):
        for dy in range(-size_search,size_search+1):
            for dz in range(-size_search,size_search+1):
                prog.run_kernel("dist",img.shape,None,
                                img,tmpImg,np.int32(dx),np.int32(dy),np.int32(dz))
                
                prog.run_kernel("convolve",img.shape,None,
                                tmpImg,tmpImg2,np.int32(1))
                prog.run_kernel("convolve",img.shape,None,
                                tmpImg2,tmpImg,np.int32(2))
                prog.run_kernel("convolve",img.shape,None,
                                tmpImg,distImg,np.int32(4))

                prog.run_kernel("computePlus",img.shape,None,
                                img,distImg,accBuf.data,weightBuf.data,
                                np.int32(img.shape[0]),
                                np.int32(img.shape[1]),
                                np.int32(img.shape[2]),
                                np.int32(dx),np.int32(dy),np.int32(dz),
                                np.float32(sigma))
                if dx!=0:
                #if any([dx,dy,dz]):
                    prog.run_kernel("computeMinus",img.shape,None,
                                    img,distImg,accBuf.data,weightBuf.data,
                                    np.int32(img.shape[0]),
                                    np.int32(img.shape[1]),
                                    np.int32(img.shape[2]),
                                    np.int32(dx),np.int32(dy),np.int32(dz),
                                    np.float32(sigma))

    accBuf /= weightBuf
    return accBuf.get()



if __name__ == '__main__':
    from time import time
    
    d = 10*np.linspace(0,1,128*129*130).reshape((128,129,130))

    d += np.random.normal(0,1,d.shape)

    t = time()
    res = nlm3(d,100,2,3)
    t = time() -t 

    print("took %.2f ms"%(1000.*t))
