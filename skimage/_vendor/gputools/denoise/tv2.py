"""
bilateral filter

"""
from __future__ import print_function, unicode_literals, absolute_import, division
import logging
logger = logging.getLogger(__name__)


import numpy as np

from gputools import OCLArray,OCLImage, OCLProgram, get_device

from ._abspath import abspath

def _tv2(data,weight,Niter=50):
    """
    chambolles tv regularized denoising

    weight should be around  2+1.5*noise_sigma
    """

    if dev is None:
        dev = imgtools.__DEFAULT_OPENCL_DEVICE__

    if dev is None:
        raise ValueError("no OpenCLDevice found...")


    proc = OCLProcessor(dev,utils.absPath("kernels/tv_chambolle.cl"))

    if Ncut ==1:
        inImg = dev.createImage(data.shape[::-1],dtype = np.float32)

        pImgs = [ dev.createImage(data.shape[::-1],
                                  mem_flags = cl.mem_flags.READ_WRITE,
                                  dtype= np.float32,
                                  channel_order = cl.channel_order.RGBA)
                                  for i in range(2)]

        outImg = dev.createImage(data.shape[::-1],
                                 dtype = np.float32,
                                 mem_flags = cl.mem_flags.READ_WRITE)


        dev.writeImage(inImg,data.astype(np.float32));
        dev.writeImage(pImgs[0],np.zeros((4,)+data.shape,dtype=np.float32));
        dev.writeImage(pImgs[1],np.zeros((4,)+data.shape,dtype=np.float32));


        for i in range(Niter):
            proc.runKernel("div_step",inImg.shape,None,
                           inImg,pImgs[i%2],outImg)
            proc.runKernel("grad_step",inImg.shape,None,
                           outImg,pImgs[i%2],pImgs[1-i%2],
                           np.float32(weight))
        return dev.readImage(outImg,dtype=np.float32)

    else:
        res = np.empty_like(data,dtype=np.float32)
        Nz,Ny,Nx = data.shape
        # a heuristic guess: Npad = Niter means perfect
        Npad = 1+Niter/2
        for i0,(i,j,k) in enumerate(product(list(range(Ncut)),repeat=3)):
            logger.info("calculating box  %i/%i"%(i0+1,Ncut**3))
            sx = slice(i*Nx/Ncut,(i+1)*Nx/Ncut)
            sy = slice(j*Ny/Ncut,(j+1)*Ny/Ncut)
            sz = slice(k*Nz/Ncut,(k+1)*Nz/Ncut)
            sx1,sx2 = utils._extended_slice(sx,Nx,Npad)
            sy1,sy2 = utils._extended_slice(sy,Ny,Npad)
            sz1,sz2 = utils._extended_slice(sz,Nz,Npad)

            data_sliced = data[sz1,sy1,sx1].copy()
            _res = tv3_gpu(dev,data_sliced,weight,Niter,Ncut = 1)
            res[sz,sy,sx] = _res[sz2,sy2,sx2]

        return res


def tv2(data,weight,Niter=50):
    """
    chambolles tv regularized denoising

    weight should be around  2+1.5*noise_sigma
    """

    prog = OCLProgram(abspath("kernels/tv2.cl"))

    data_im = OCLImage.from_array(data.astype(np,float32,copy=False))

    pImgs = [ dev.createImage(data.shape[::-1],
                                  mem_flags = cl.mem_flags.READ_WRITE,
                                  dtype= np.float32,
                                  channel_order = cl.channel_order.RGBA)
                                  for i in range(2)]

    outImg = dev.createImage(data.shape[::-1],
                             dtype = np.float32,
                             mem_flags = cl.mem_flags.READ_WRITE)


    dev.writeImage(inImg,data.astype(np.float32));
    dev.writeImage(pImgs[0],np.zeros((4,)+data.shape,dtype=np.float32));
    dev.writeImage(pImgs[1],np.zeros((4,)+data.shape,dtype=np.float32));


    for i in range(Niter):
        proc.runKernel("div_step",inImg.shape,None,
                           inImg,pImgs[i%2],outImg)
        proc.runKernel("grad_step",inImg.shape,None,
                           outImg,pImgs[i%2],pImgs[1-i%2],
                           np.float32(weight))
    return dev.readImage(outImg,dtype=np.float32)



if __name__ == '__main__':
    from scipy.misc import lena

    d = lena()

    d = np.random.poisson(d,d.shape)

    res = bilateral2(d,3,100.)
