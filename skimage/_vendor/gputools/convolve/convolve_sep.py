from __future__ import print_function, unicode_literals, absolute_import, division
import logging

logger = logging.getLogger(__name__)

import os
import numpy as np
from gputools import OCLArray, OCLProgram, get_device

from gputools.core.ocltypes import assert_bufs_type
from gputools.utils.tile_iterator import tile_iterator
from gputools.separable import separable_series

from ._abspath import abspath


def convolve_sep2(data, hx, hy, res_g=None, sub_blocks=None):
    """convolves 2d data with kernel h = outer(hx,hy)
    boundary conditions are clamping to edge.

    data is either np array or a gpu buffer (OCLArray)

    """

    if isinstance(data, np.ndarray):
        data = np.ascontiguousarray(data)

        if sub_blocks == (1, 1) or sub_blocks is None:
            return _convolve_sep2_numpy(data, hx, hy)
        else:
            # cut the image into tile and operate on every of them
            N_sub = [int(np.ceil(1. * n / s)) for n, s in zip(data.shape, sub_blocks)]
            Npads = [int(len(_h) / 2) for _h in [hy, hx]]
            res = np.empty(data.shape, np.float32)
            for data_tile, data_s_src, data_s_dest \
                    in tile_iterator(data, blocksize=N_sub,
                                     padsize=Npads,
                                     mode="constant"):
                res_tile = _convolve_sep2_numpy(data_tile.copy(),
                                                hx, hy)
                res[data_s_src] = res_tile[data_s_dest]
            return res
    elif isinstance(data, OCLArray):
        if not sub_blocks is None:
            raise NotImplementedError()
        return _convolve_sep2_gpu(data, hx, hy, res_g=res_g)
    else:
        raise TypeError("array argument (1) has bad type: %s" % type(data))


def _convolve_sep2_numpy(data, hx, hy):
    hx_g = OCLArray.from_array(hx.astype(np.float32))
    hy_g = OCLArray.from_array(hy.astype(np.float32))

    data_g = OCLArray.from_array(data.astype(np.float32))

    return _convolve_sep2_gpu(data_g, hx_g, hy_g).get()


def _convolve_sep2_gpu(data_g, hx_g, hy_g, res_g=None):
    assert_bufs_type(np.float32, data_g, hx_g, hy_g)

    prog = OCLProgram(abspath("kernels/convolve_sep.cl"))

    Ny, Nx = hy_g.shape[0], hx_g.shape[0]

    tmp_g = OCLArray.empty_like(data_g)

    if res_g is None:
        res_g = OCLArray.empty_like(data_g)

    prog.run_kernel("conv_sep2_x", data_g.shape[::-1], None, data_g.data, hx_g.data, tmp_g.data, np.int32(Nx))
    prog.run_kernel("conv_sep2_y", data_g.shape[::-1], None, tmp_g.data, hy_g.data, res_g.data, np.int32(Ny))

    return res_g


def convolve_sep3(data, hx, hy, hz, res_g=None, sub_blocks=(1, 1, 1), tmp_g = None):
    """convolves 3d data with kernel h = outer(hx,hy, hz)
    boundary conditions are clamping to edge.

    data, hx, hy.... are either np array or a gpu buffer (OCLArray)

    """

    if isinstance(data, np.ndarray):
        data = np.ascontiguousarray(data)
        if sub_blocks == (1, 1, 1) or sub_blocks is None:
            return _convolve_sep3_numpy(data, hx, hy, hz)
        else:
            # cut the image into tile and operate on every of them
            N_sub = [int(np.ceil(1. * n / s)) for n, s in zip(data.shape, sub_blocks)]
            Npads = [int(len(_h) / 2) for _h in [hz, hy, hx]]
            res = np.empty(data.shape, np.float32)
            for i, (data_tile, data_s_src, data_s_dest) \
                    in enumerate(tile_iterator(data, blocksize=N_sub,
                                               padsize=Npads,
                                               mode="constant")):
                res_tile = _convolve_sep3_numpy(data_tile.copy(),
                                                hx, hy, hz)
                res[data_s_src] = res_tile[data_s_dest]
            return res


    elif isinstance(data, OCLArray):
        return _convolve_sep3_gpu(data, hx, hy, hz, res_g=res_g, tmp_g = tmp_g)
    else:
        raise TypeError("array argument (1) has bad type: %s" % type(data))


def _convolve_sep3_numpy(data, hx, hy, hz):
    hx_g = OCLArray.from_array(hx.astype(np.float32))
    hy_g = OCLArray.from_array(hy.astype(np.float32))
    hz_g = OCLArray.from_array(hz.astype(np.float32))

    data_g = OCLArray.from_array(data.astype(np.float32))

    return _convolve_sep3_gpu(data_g, hx_g, hy_g, hz_g).get()


def _convolve_sep3_gpu(data_g, hx_g, hy_g, hz_g, res_g=None, tmp_g = None):
    assert_bufs_type(np.float32, data_g, hx_g, hy_g)

    prog = OCLProgram(abspath("kernels/convolve_sep.cl"))

    Nz, Ny, Nx = hz_g.shape[0], hy_g.shape[0], hx_g.shape[0]

    if tmp_g is None:
        tmp_g = OCLArray.empty_like(data_g)

    if res_g is None:
        res_g = OCLArray.empty_like(data_g)

    prog.run_kernel("conv_sep3_x", data_g.shape[::-1], None, data_g.data, hx_g.data, res_g.data, np.int32(Nx))
    prog.run_kernel("conv_sep3_y", data_g.shape[::-1], None, res_g.data, hy_g.data, tmp_g.data, np.int32(Ny))
    prog.run_kernel("conv_sep3_z", data_g.shape[::-1], None, tmp_g.data, hz_g.data, res_g.data, np.int32(Nz))

    return res_g


def convolve_sep_approx(data, h, Nsep=2):
    if data.ndim == 2:
        sep_func = convolve_sep2
    elif data.ndim == 3:
        sep_func = convolve_sep3
    else:
        raise NotImplementedError("data has to 2 or 3 dimensional!")

    hs = separable_series(h + 1.e-40, Nsep)
    res = np.zeros(data.shape, np.float32)

    for i, h in enumerate(hs):
        print("sep blur %s/%s   (%s)" % (i + 1, len(hs), np.prod([np.sum(_h) for _h in h])))
        res += sep_func(data, *h[::-1])

    return res


def test_2d():
    import time

    data = np.zeros((100,) * 2, np.float32)

    data[50, 50] = 1.
    hx = 1. / 5 * np.ones(5)
    hy = 1. / 13 * np.ones(13)

    out = convolve_sep2(data, hx, hy)

    data_g = OCLArray.from_array(data.astype(np.float32))
    hx_g = OCLArray.from_array(hx.astype(np.float32))
    hy_g = OCLArray.from_array(hy.astype(np.float32))

    out_g = convolve_sep2(data_g, hx_g, hy_g)

    return out, out_g.get()


def test_3d():
    from time import time
    Niter = 10

    data = np.zeros((128,) * 3, np.float32)

    data[30, 30, 30] = 1.
    hx = 1. / 5 * np.ones(5)
    hy = 1. / 13 * np.ones(13)
    hz = 1. / 13 * np.ones(11)

    t = time()
    for _ in range(Niter):
        out = convolve_sep3(data, hx, hy, hz)
    print("time: %.3f ms" % (1000. * (time() - t) / Niter))

    data_g = OCLArray.from_array(data.astype(np.float32))
    hx_g = OCLArray.from_array(hx.astype(np.float32))
    hy_g = OCLArray.from_array(hy.astype(np.float32))
    hz_g = OCLArray.from_array(hz.astype(np.float32))

    t = time()
    for _ in range(Niter):
        out_g = convolve_sep3(data_g, hx_g, hy_g, hz_g)

    out_g.get();
    print("time: %.3f ms" % (1000. * (time() - t) / Niter))

    return out, out_g.get()


if __name__ == '__main__':
    # out1, out2 = test_3d()
    out1, out2 = test_2d()
