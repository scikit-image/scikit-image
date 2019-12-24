from __future__ import print_function, unicode_literals, absolute_import, division
import logging

logger = logging.getLogger(__name__)

import numpy as np
import warnings
from mako.template import Template

from gputools import OCLArray, OCLProgram, tile_iterator
from gputools.core.ocltypes import cl_buffer_datatype_dict
from ._abspath import abspath



def _median_filter_gpu_2d():
    def _filt(data_g, size=(3, 3), cval = 0,  res_g=None):
        if not data_g.dtype.type in cl_buffer_datatype_dict:
            raise ValueError("dtype %s not supported" % data_g.dtype.type)

        DTYPE = cl_buffer_datatype_dict[data_g.dtype.type]


        with open(abspath("kernels/median_filter.cl"), "r") as f:
            tpl = Template(f.read())


        rendered = tpl.render(DTYPE = DTYPE,FSIZE_X=size[-1], FSIZE_Y=size[-2], FSIZE_Z=1, CVAL = cval)


        prog = OCLProgram(src_str=rendered)

        tmp_g = OCLArray.empty_like(data_g)

        if res_g is None:
            res_g = OCLArray.empty_like(data_g)

        prog.run_kernel("median_2", data_g.shape[::-1], None, data_g.data, res_g.data)
        return res_g

    return _filt


def _median_filter_gpu_3d():
    def _filt(data_g, size=(3, 3, 3), cval = 0, res_g=None):
        if not data_g.dtype.type in cl_buffer_datatype_dict:
            raise ValueError("dtype %s not supported" % data_g.dtype.type)

        DTYPE = cl_buffer_datatype_dict[data_g.dtype.type]


        with open(abspath("kernels/median_filter.cl"), "r") as f:
            tpl = Template(f.read())

        rendered = tpl.render(DTYPE = DTYPE,FSIZE_X=size[2], FSIZE_Y=size[1], FSIZE_Z=size[0],CVAL = cval)

        prog = OCLProgram(src_str=rendered)

        tmp_g = OCLArray.empty_like(data_g)

        if res_g is None:
            res_g = OCLArray.empty_like(data_g)

        prog.run_kernel("median_3", data_g.shape[::-1], None, data_g.data, res_g.data)
        return res_g

    return _filt


####################################################################################



def make_filter(filter_gpu):
    def _filter_numpy(data, size, cval):
        if not data.dtype.type in cl_buffer_datatype_dict:
            warnings.warn("%s data not supported, casting to np.float32"%data.dtype.type )
            data = data.astype(np.float32)
        data_g = OCLArray.from_array(data)
        return filter_gpu(data_g = data_g, size=size, cval = cval).get()

    def _filter(data, size=10, cval = 0, res_g=None, sub_blocks=(1, 1, 1)):
        if np.isscalar(size):
            size = (size,)*len(data.shape)

        if isinstance(data, np.ndarray):
            if sub_blocks is None or set(sub_blocks) == {1}:
                return _filter_numpy(data, size, cval)
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
                                                 size, cval)
                    res[data_s_src] = res_tile[data_s_dest]
                return res

        elif isinstance(data, OCLArray):
            return filter_gpu(data, size=size, cval = cval, res_g=res_g)
        else:
            raise TypeError("array argument (1) has bad type: %s" % type(data))

    return _filter



def median_filter(data, size=3, cval = 0, res_g=None, sub_blocks=None):
    """
        median filter of given size

    Parameters
    ----------
    data: 2 or 3 dimensional ndarray or OCLArray of type float32
        input data
    size: scalar, tuple
        the size of the patch to consider
    cval: scalar, 
        the constant value for out of border access (cf mode = "constant")
    
    res_g: OCLArray
        store result in buffer if given
    sub_blocks:
        perform over subblock tiling (only if data is ndarray)

    Returns
    -------
        filtered image or None (if OCLArray)
    """
    if data.ndim == 2:
        _filt = make_filter(_median_filter_gpu_2d())
    elif data.ndim == 3:
        _filt = make_filter(_median_filter_gpu_3d())
    else:
        raise ValueError("currently only 2 or 3 dimensional data is supported")

    return _filt(data=data, size=size, cval = cval, res_g=res_g, sub_blocks=sub_blocks)
