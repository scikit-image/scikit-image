from __future__ import print_function, unicode_literals, absolute_import, division
import logging

logger = logging.getLogger(__name__)


import numpy as np
import warnings
from mako.template import Template

from gputools import OCLArray, OCLProgram, OCLElementwiseKernel
from gputools.core.ocltypes import assert_bufs_type, cl_buffer_datatype_dict
from gputools.utils.tile_iterator import tile_iterator
from ._abspath import abspath


def _generic_filter_gpu_2d(FUNC = "fmax(res,val)", DEFAULT = "-INFINITY"):
    def _filt(data_g, size=(3, 3), res_g=None):
        if not data_g.dtype.type in cl_buffer_datatype_dict:
            raise ValueError("dtype %s not supported"%data_g.dtype.type)

        DTYPE = cl_buffer_datatype_dict[data_g.dtype.type]

        with open(abspath("kernels/generic_separable_filter.cl"), "r") as f:
            tpl = Template(f.read())


        rendered = tpl.render(FSIZE_X=size[-1], FSIZE_Y=size[-2], FSIZE_Z=1,
                              FUNC=FUNC, DEFAULT=DEFAULT, DTYPE = DTYPE)

        prog = OCLProgram(src_str=rendered)

        tmp_g = OCLArray.empty_like(data_g)

        if res_g is None:
            res_g = OCLArray.empty_like(data_g)

        prog.run_kernel("filter_2_x", data_g.shape[::-1], None, data_g.data, tmp_g.data)
        prog.run_kernel("filter_2_y", data_g.shape[::-1], None, tmp_g.data, res_g.data)
        return res_g
    return _filt

def _generic_filter_gpu_3d(FUNC = "fmax(res,val)", DEFAULT = "-INFINITY"):
    def _filt(data_g, size=(3, 3,3 ), res_g=None):
        if not data_g.dtype.type in cl_buffer_datatype_dict:
            raise ValueError("dtype %s not supported"%data_g.dtype.type)

        DTYPE = cl_buffer_datatype_dict[data_g.dtype.type]


        with open(abspath("kernels/generic_separable_filter.cl"), "r") as f:
            tpl = Template(f.read())

        rendered = tpl.render(FSIZE_X=size[-1], FSIZE_Y=size[-2], FSIZE_Z=size[-3],
                              FUNC=FUNC, DEFAULT=DEFAULT, DTYPE = DTYPE)

        prog = OCLProgram(src_str=rendered,
                          build_options = ["-cl-unsafe-math-optimizations"]
        )                       

        tmp_g = OCLArray.empty_like(data_g)

        if res_g is None:
            res_g = OCLArray.empty_like(data_g)

        prog.run_kernel("filter_3_x", data_g.shape[::-1], None, data_g.data, res_g.data)
        prog.run_kernel("filter_3_y", data_g.shape[::-1], None, res_g.data, tmp_g.data)
        prog.run_kernel("filter_3_z", data_g.shape[::-1], None, tmp_g.data, res_g.data)
        return res_g
    return _filt



def make_filter(filter_gpu):
    def _filter_numpy(data, size):
        if not data.dtype.type in cl_buffer_datatype_dict:
            warnings.warn("%s data not supported, casting to np.float32"%data.dtype.type )
            data = data.astype(np.float32)
        data_g = OCLArray.from_array(data)
        return filter_gpu(data_g = data_g, size=size).get()

    def _filter(data, size=10, res_g=None, sub_blocks=(1, 1, 1)):
        if np.isscalar(size):
            size = (size,)*len(data.shape)

        if isinstance(data, np.ndarray):
            if sub_blocks is None or set(sub_blocks) == {1}:
                return _filter_numpy(data, size)
            else:
                # cut the image into tile and operate on every of them
                N_sub = [int(np.ceil(1. * n / s)) for n, s in zip(data.shape, sub_blocks)]
                Npads = int(size // 2)
                res = np.empty(data.shape, np.float32)
                for i, (data_tile, data_s_src, data_s_dest) \
                        in enumerate(tile_iterator(data, blocksize=N_sub,
                                                   padsize=Npads,
                                                   mode="constant")):
                    res_tile = _filter_numpy(data_tile.copy(),
                                                 size)
                    res[data_s_src] = res_tile[data_s_dest]
                return res

        elif isinstance(data, OCLArray):
            return filter_gpu(data, size=size, res_g=res_g)
        else:
            raise TypeError("array argument (1) has bad type: %s" % type(data))

    return _filter


####################################################################################



def max_filter(data, size=7, res_g=None, sub_blocks=(1, 1, 1)):
    """
        maximum filter of given size

    Parameters
    ----------
    data: 2 or 3 dimensional ndarray or OCLArray of type float32
        input data
    size: scalar, tuple
        the size of the patch to consider
    res_g: OCLArray
        store result in buffer if given
    sub_blocks:
        perform over subblock tiling (only if data is ndarray)

    Returns
    -------
        filtered image or None (if OCLArray)
    """
    if data.ndim == 2:
        _filt = make_filter(_generic_filter_gpu_2d(FUNC = "(val>res?val:res)", DEFAULT = "-INFINITY"))
    elif data.ndim == 3:
        _filt = make_filter(_generic_filter_gpu_3d(FUNC = "(val>res?val:res)", DEFAULT = "-INFINITY"))

    return _filt(data = data, size = size, res_g = res_g, sub_blocks=sub_blocks)


def min_filter(data, size=7, res_g=None, sub_blocks=(1, 1, 1)):
    """
        minimum filter of given size

    Parameters
    ----------
    data: 2 or 3 dimensional ndarray or OCLArray of type float32
        input data
    size: scalar, tuple
        the size of the patch to consider
    res_g: OCLArray
        store result in buffer if given
    sub_blocks:
        perform over subblock tiling (only if data is ndarray)

    Returns
    -------
        filtered image or None (if OCLArray)
    """
    if data.ndim == 2:
        _filt = make_filter(_generic_filter_gpu_2d(FUNC="(val<res?val:res)", DEFAULT="INFINITY"))
    elif data.ndim == 3:
        _filt = make_filter(_generic_filter_gpu_3d(FUNC="(val<res?val:res)", DEFAULT="INFINITY"))
    else:
        raise ValueError("currently only 2 or 3 dimensional data is supported")
    return _filt(data=data, size=size, res_g=res_g, sub_blocks=sub_blocks)



def uniform_filter(data, size=7, res_g=None, sub_blocks=(1, 1, 1), normalized = True):
    """
        mean filter of given size

    Parameters
    ----------
    data: 2 or 3 dimensional ndarray or OCLArray of type float32
        input data
    size: scalar, tuple
        the size of the patch to consider
    res_g: OCLArray
        store result in buffer if given
    sub_blocks:
        perform over subblock tiling (only if data is ndarray)
    normalized: bool
        if True, the filter corresponds to mean
        if False, the filter corresponds to sum

    Returns
    -------
        filtered image or None (if OCLArray)
    """

    if normalized:
        if np.isscalar(size):
            norm = size
        else:
            norm = np.int32(np.prod(size))**(1./len(size))
        FUNC = "res+val/%s"%norm
    else:
        FUNC = "res+val"

    if data.ndim == 2:
        _filt = make_filter(_generic_filter_gpu_2d(FUNC=FUNC, DEFAULT="0"))
    elif data.ndim == 3:
        _filt = make_filter(_generic_filter_gpu_3d(FUNC=FUNC, DEFAULT="0"))

    res =  _filt(data=data, size=size, res_g=res_g, sub_blocks=sub_blocks)

    return res




# FIXME: only to compare aganst gputools.gaussian_flter (which uses convolve_sep)
def _gauss_filter(data, sigma=4, res_g=None, sub_blocks=(1, 1, 1)):
    """
        gaussian filter of given size

    Parameters
    ----------
    data: 2 or 3 dimensional ndarray or OCLArray of type float32
        input data
    size: scalar, tuple
        the size of the patch to consider
    res_g: OCLArray
        store result in buffer if given
    sub_blocks:
        perform over subblock tiling (only if data is ndarray)

    Returns
    -------
        filtered image or None (if OCLArray)
    """
    truncate = 4.
    radius = tuple(int(truncate*s +0.5) for s in sigma)
    size = tuple(2*r+1 for r in radius)

    s = sigma[0]

    if data.ndim == 2:
        _filt = make_filter(_generic_filter_gpu_2d(FUNC="res+(val*native_exp((float)(-(ht-%s)*(ht-%s)/2/%s/%s)))"%(size[0]//2,size[0]//2,s,s), DEFAULT="0.f"))
    elif data.ndim == 3:
        _filt = make_filter(_generic_filter_gpu_3d(FUNC="res+(val*native_exp((float)(-(ht-%s)*(ht-%s)/2/%s/%s)))"%(size[0]//2,size[0]//2,s,s), DEFAULT="0.f"))

    else:
        raise ValueError("currently only 2 or 3 dimensional data is supported")
    return _filt(data=data, size=size, res_g=res_g, sub_blocks=sub_blocks)



