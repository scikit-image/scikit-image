# from __future__ import print_function, unicode_literals, absolute_import, division
# import logging
# logger = logging.getLogger(__name__)
#
# import numpy as np
# from PyOCL import OCLDevice, OCLProcessor, cl
#
# from convolve import convolve2
#
# from _abspath import abspath
#
# def _correlate2(data,h, dev = None):
#     """computes normalized cross correlation of 2d <data> with template <h> on the GPU Device <dev>
#     boundary conditions are clamping to edge.
#     h is converted to float32
#     if dev == None a new one is created
#
#     """
#
#     if not dev:
#         dev = OCLDevice(useDevice = imgtools.__OPENCLDEVICE__)
#
#     dtype = data.dtype.type
#
#     dtypes_kernels = {np.float32:"correlate2d_float",
#                       np.uint16:"correlate2d_short"}
#
#     if not dtype in dtypes_kernels.keys():
#         raise TypeError("data type %s not supported yet, please convert to:"%dtype,dtypes_kernels.keys())
#
#
#     proc = OCLProcessor(dev,absPath("kernels/correlate_kernels.cl"))
#
#     Ny,Nx = h.shape
#
#     hbuf = dev.createBuffer(Nx*Ny,dtype=np.float32,mem_flags= cl.mem_flags.READ_ONLY)
#     inImg = dev.createImage_like(data)
#     outImg = dev.createImage_like(data,mem_flags="READ_WRITE")
#
#     dev.writeImage(inImg,data)
#
#     dev.writeBuffer(hbuf,h.astype(np.float32).flatten())
#
#     proc.runKernel(dtypes_kernels[dtype],inImg.shape,None,inImg,hbuf,np.int32(Nx),np.int32(Ny),outImg)
#
#
#     return dev.readImage(outImg)
#
#
# def correlate2(data,h, dev = None):
#     """computes normalized cross correlation of 2d <data> with template <h> on the GPU Device <dev>
#     boundary conditions are clamping to edge.
#     h is converted to float32
#     if dev == None a new one is created
#
#     """
#
#     if not dev:
#         dev = OCLDevice(useDevice = imgtools.__OPENCLDEVICE__)
#
#
#     # normalize data and template
#     #data
#
#     dtype = data.dtype.type
#
#     dtypes_kernels = {np.float32:"mean_var_2d_float",
#                       np.uint16:"mean_var_2d_short"}
#
#     if not dtype in dtypes_kernels.keys():
#         raise TypeError("data type %s not supported yet, please convert to:"%dtype,dtypes_kernels.keys())
#
#
#     proc = OCLProcessor(dev,absPath("kernels/correlate_kernels.cl"))
#
#     Ny,Nx = h.shape
#
#     inImg = dev.createImage_like(data)
#     meanBuf = dev.createBuffer(data.size,dtype=dtype,mem_flags= cl.mem_flags.READ_WRITE)
#     varBuf = dev.createBuffer(data.size,dtype=dtype,mem_flags= cl.mem_flags.READ_WRITE)
#
#     dev.writeImage(inImg,data)
#
#     proc.runKernel(dtypes_kernels[dtype],inImg.shape,None,inImg,np.int32(data.shape[1]),np.int32(Nx),np.int32(Ny),meanBuf,varBuf)
#
#     dataMean, dataVar = dev.readBuffer(meanBuf,dtype).reshape(data.shape),dev.readBuffer(varBuf,dtype).reshape(data.shape)
#
#     #template
#     hMean = 1.*np.mean(h.flatten())
#     hVar = 1.*np.var(h.flatten())
#
#     res = convolve2(dev,data-dataMean,h - hMean)
#
#     print hMean, hVar
#     # # res = convolve2(dev,data-dataMean,h)
#
#     # return res
#     # return dataVar
#     return res/np.maximum(1.e-6,np.sqrt(dataVar*hVar))
#
#
#     # return dataMean, dataVar
#
#
#
#
#
# if __name__ == '__main__':
#
#     N, Nh = 64, 11
#     data = np.zeros((N,)*2,np.float32)
#
#     t = np.linspace(-1,1,Nh)
#
#     hx = np.exp(-3*t**2)
#     hy = np.exp(-3*t**2)
#
#     h = np.outer(hy,hx)
#
#     # h = np.ones((Nh,)*2)
#
#     data[10:10+Nh,1:1+Nh] = 1.*h
#
#
#     dev = OCLDevice()
#     out = correlate2(data,h, dev)
