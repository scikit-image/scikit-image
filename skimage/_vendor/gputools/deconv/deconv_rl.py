""" Lucy richardson deconvolution
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import logging

logger = logging.getLogger(__name__)

import os
import numpy as np
from gputools import OCLArray, OCLProgram, get_device
from gputools import convolve, fft_convolve, fft, fft_plan
from gputools import OCLElementwiseKernel
from ._abspath import abspath

_multiply_inplace = OCLElementwiseKernel(
    "float *a, float * b",
    "a[i] = a[i] * b[i]",
    "mult_inplace")

_divide_inplace = OCLElementwiseKernel(
    "float *a, float * b",
    "b[i] = a[i]*b[i]/(b[i]*b[i]+0.001f)",
    "divide_inplace")

_complex_multiply = OCLElementwiseKernel(
    "cfloat_t *a, cfloat_t * b,cfloat_t * res",
    "res[i] = cfloat_mul(a[i],b[i])",
    "mult")

_complex_multiply_inplace = OCLElementwiseKernel(
    "cfloat_t *a, cfloat_t * b",
    "a[i] = cfloat_mul(a[i],b[i])",
    "mult_inplace")

_complex_divide = OCLElementwiseKernel(
    "cfloat_t *a, cfloat_t * b,cfloat_t * res",
    "res[i] = cfloat_divide(b[i],a[i])",
    "div")

_complex_divide_inplace = OCLElementwiseKernel(
    "cfloat_t *a, cfloat_t * b",
    "b[i] = cfloat_divide(a[i],b[i])",
    "divide_inplace")


def deconv_rl(data, h, Niter=10, mode_conv="fft", log_iter=False):
    """ richardson lucy deconvolution of data with psf h
    using spatial convolutions (h should be small then)

    mode_conv = "fft" or "spatial"
    """

    if isinstance(data, np.ndarray):
        mode_data = "np"
    elif isinstance(data, OCLArray):
        mode_data = "gpu"
    else:
        raise TypeError("array argument (1) has bad type: %s"%type(arr_obj))

    if mode_conv=="fft":
        res = _deconv_rl_np_fft(data, h, Niter) if mode_data=="np" else _deconv_rl_gpu_fft(data, h, Niter)
        return res
    elif mode_conv=="spatial":
        res = _deconv_rl_np_conv(data, h, Niter) if mode_data=="np" else gpu_conv(data, h, Niter)
        return res
    else:
        raise KeyError("mode_conv = %s not known, either 'fft' or 'spatial'"%mode_conv)


def _deconv_rl_np_conv(data, h, Niter=10, ):
    """
    """
    d_g = OCLArray.from_array(data.astype(np.float32, copy=False))
    h_g = OCLArray.from_array(h.astype(np.float32, copy=False))
    res_g = _deconv_rl_gpu_conv(d_g, h_g, Niter)
    return res_g.get()



def _deconv_rl_gpu_conv(data_g, h_g, Niter=10):
    """
    using convolve

    """

    # set up some gpu buffers
    u_g = OCLArray.empty(data_g.shape, np.float32)

    u_g.copy_buffer(data_g)

    tmp_g = OCLArray.empty(data_g.shape, np.float32)
    tmp2_g = OCLArray.empty(data_g.shape, np.float32)


    # fix this
    hflip_g = OCLArray.from_array((h_g.get()[::-1, ::-1]).copy())



    for i in range(Niter):
        convolve(u_g, h_g,
                 res_g=tmp_g)

        _divide_inplace(data_g, tmp_g)

        # return data_g, tmp_g

        convolve(tmp_g, hflip_g,
                 res_g=tmp2_g)



        _multiply_inplace(u_g, tmp2_g)

    return u_g




def _deconv_rl_np_fft(data, h, Niter=10,
                      h_is_fftshifted=False):
    """ deconvolves data with given psf (kernel) h

    data and h have to be same shape


    via lucy richardson deconvolution
    """

    if data.shape!=h.shape:
        raise ValueError("data and h have to be same shape")

    if not h_is_fftshifted:
        h = np.fft.fftshift(h)

    hflip = h[::-1, ::-1]

    # set up some gpu buffers
    y_g = OCLArray.from_array(data.astype(np.complex64))
    u_g = OCLArray.from_array(data.astype(np.complex64))

    tmp_g = OCLArray.empty(data.shape, np.complex64)

    hf_g = OCLArray.from_array(h.astype(np.complex64))
    hflip_f_g = OCLArray.from_array(hflip.astype(np.complex64))

    # hflipped_g = OCLArray.from_array(h.astype(np.complex64))

    plan = fft_plan(data.shape)

    # transform psf
    fft(hf_g, inplace=True)
    fft(hflip_f_g, inplace=True)

    for i in range(Niter):
        logger.info("Iteration: {}".format(i))
        fft_convolve(u_g, hf_g,
                     res_g=tmp_g,
                     kernel_is_fft=True)

        _complex_divide_inplace(y_g, tmp_g)

        fft_convolve(tmp_g, hflip_f_g,
                     inplace=True,
                     kernel_is_fft=True)

        _complex_multiply_inplace(u_g, tmp_g)

    return np.abs(u_g.get())


def _deconv_rl_gpu_fft(data_g, h_g, Niter=10):
    """
    using fft_convolve

    """

    if data_g.shape!=h_g.shape:
        raise ValueError("data and h have to be same shape")

    # set up some gpu buffers
    u_g = OCLArray.empty(data_g.shape, np.complex64)

    u_g.copy_buffer(data_g)

    tmp_g = OCLArray.empty(data_g.shape, np.complex64)

    # fix this
    hflip_g = OCLArray.from_array((h_g.get()[::-1, ::-1]).copy())

    plan = fft_plan(data_g.shape)

    # transform psf
    fft(h_g, inplace=True)
    fft(hflip_g, inplace=True)

    for i in range(Niter):
        logger.info("Iteration: {}".format(i))
        fft_convolve(u_g, h_g,
                     res_g=tmp_g,
                     kernel_is_fft=True)

        _complex_divide_inplace(data_g, tmp_g)

        fft_convolve(tmp_g, hflip_g,
                     inplace=True,
                     kernel_is_fft=True)

        _complex_multiply_inplace(u_g, tmp_g)

    return u_g


if __name__=='__main__':
    from scipy.misc import face
    from gputools import pad_to_shape

    d = np.pad(face()[200:642, 400:842, 0].astype(np.float32),((35,)*2,)*2, mode = "constant")

    h = np.ones((31,)*2)
    h *= 1./np.sum(h)
    h = pad_to_shape(h,d.shape)

    y = fft_convolve(d, h)

    y += 0.1*np.max(d)*np.random.uniform(0, 1, d.shape)

    print("start")

    u = deconv_rl(y, h, 20, mode_conv="fft")
