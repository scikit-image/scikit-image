from __future__ import print_function, unicode_literals, absolute_import, division
import logging

logger = logging.getLogger(__name__)

import numpy as np

from gputools import OCLProgram, OCLArray, OCLImage, get_device
from gputools.core.ocltypes import assert_bufs_type
from gputools.utils.tile_iterator import tile_iterator

import pyopencl as cl
from ._abspath import abspath


def convolve(data, h, res_g=None, sub_blocks=None, mode='constant'):
    """
    convolves 1d-3d data with kernel h 

    data and h can either be numpy arrays or gpu buffer objects (OCLArray, 
    which must be float32 then)

    boundary conditions are clamping to zero at edge.
    
    """
    sub_blocks = sub_blocks or (1,) * data.ndim

    if not len(data.shape) in [1, 2, 3]:
        raise ValueError("dim = %s not supported" % (len(data.shape)))

    if len(data.shape) != len(h.shape):
        raise ValueError("dimemnsion of data (%s) and h (%s) are different" % (len(data.shape), len(h.shape)))

    if isinstance(data, OCLArray) and isinstance(h, OCLArray):
        return _convolve_buf(data, h, res_g)
    elif isinstance(data, np.ndarray) and isinstance(h, np.ndarray):
        if sub_blocks == (1,) * data.ndim and mode == 'constant':
            res = _convolve_np(data, h)
        else:
            # cut the image into tile and operate on every of them
            N_sub = [int(np.ceil(1. * n / s)) for n, s in zip(data.shape, sub_blocks)]
            Npads = [int(s / 2) for s in h.shape]
            res = np.empty(data.shape, np.float32)
            for data_tile, data_s_src, data_s_dest \
                    in tile_iterator(data, blocksize=N_sub,
                                     padsize=Npads,
                                     mode=mode):
                res_tile = _convolve_np(data_tile.copy(),
                                        h)
                res[data_s_src] = res_tile[data_s_dest]
        return res
    else:
        raise TypeError("unknown types (%s, %s)" % (type(data), type(h)))


def _convolve_np(data, h):
    """
    numpy variant
    """

    data_g = OCLArray.from_array(np.require(data,np.float32,"C"))
    h_g = OCLArray.from_array(np.require(h,np.float32,"C"))

    return _convolve_buf(data_g, h_g).get()


def _convolve_buf(data_g, h_g, res_g=None):
    """
    buffer variant
    """
    assert_bufs_type(np.float32, data_g, h_g)

    prog = OCLProgram(abspath("kernels/convolve.cl"))

    if res_g is None:
        res_g = OCLArray.empty(data_g.shape, dtype=np.float32)

    Nhs = [np.int32(n) for n in h_g.shape]

    kernel_name = "convolve%sd_buf" % (len(data_g.shape))


    try:
        prog.run_kernel(kernel_name, data_g.shape[::-1], None,
                        data_g.data, h_g.data, res_g.data,
                        *Nhs)

    except cl.cffi_cl.LogicError as e:
        # this catches the logic error if the kernel is to big for constant memory
        if e.code == -52:
            kernel_name = "convolve%sd_buf_global" % (len(data_g.shape))
            prog.run_kernel(kernel_name, data_g.shape[::-1], None,
                            data_g.data, h_g.data, res_g.data,
                            *Nhs)

        else:
            raise e
    except cl.cffi_cl.RuntimeError as e:
        # this catches the runtime error if the kernel is to big for constant memory
        if e.code == -5:
            kernel_name = "convolve%sd_buf_global" % (len(data_g.shape))
            prog.run_kernel(kernel_name, data_g.shape[::-1], None,
                            data_g.data, h_g.data, res_g.data,
                            *Nhs)

        else:
            raise e




    return res_g


def _convolve3_old(data, h, dev=None):
    """convolves 3d data with kernel h on the GPU Device dev
    boundary conditions are clamping to edge.
    h is converted to float32

    if dev == None the default one is used
    """

    if dev is None:
        dev = get_device()

    if dev is None:
        raise ValueError("no OpenCLDevice found...")

    dtype = data.dtype.type

    dtypes_options = {np.float32: "",
                      np.uint16: "-D SHORTTYPE"}

    if not dtype in dtypes_options:
        raise TypeError("data type %s not supported yet, please convert to:" % dtype, list(dtypes_options.keys()))

    prog = OCLProgram(abspath("kernels/convolve3.cl"),
                      build_options=dtypes_options[dtype])

    hbuf = OCLArray.from_array(h.astype(np.float32))
    img = OCLImage.from_array(data)
    res = OCLArray.empty(data.shape, dtype=np.float32)

    Ns = [np.int32(n) for n in data.shape + h.shape]

    prog.run_kernel("convolve3d", img.shape, None,
                    img, hbuf.data, res.data,
                    *Ns)

    return res.get()


def test_convolve():
    from time import time

    data = np.ones((100, 120, 140))
    h = np.ones((10, 11, 12))

    # out = convolve(data,h)
    out = convolve(data[0, ...], h[0, ...])
    out = convolve(data[0, 0, ...], h[0, 0, ...])


if __name__ == '__main__':
    # test_convolve()

    N = 100
    ndim = 3

    d = np.zeros([N + 3 * i for i, n in enumerate(range(ndim))], np.float32)
    h = np.ones((11,) * ndim, np.float32)

    ind = [np.random.randint(0, n, int(np.prod(d.shape) ** (1. / d.ndim)) / 10) for n in d.shape]
    d[tuple(ind)] = 1.
    h *= 1. / np.sum(h)

    out1 = convolve(d, h)

    d_g = OCLArray.from_array(d)
    h_g = OCLArray.from_array(h)

    res_g = convolve(d_g, h_g)

    out2 = res_g.get()
