"""

mweigert@mpi-cbg.de
"""
from __future__ import print_function, unicode_literals, absolute_import, division
import numpy as np
import math
import pyopencl as cl
from gputools import OCLProgram, OCLArray, get_device
from gputools.utils import next_power_of_2
from gputools.core.ocltypes import assert_bufs_type, cl_buffer_datatype_dict
from ._abspath import abspath

_output_type_dict = {
    np.float32: np.float32,
    np.uint8: np.uint64,
    np.uint16: np.uint64,
    np.int32: np.int64,
}


def integral_image(x, res_g=None, tmp_g=None):
    """
    Computes the (inclusive) integral image of a 2D or 3D array  
    
    x can be a ndarray or a OCLArray
    
    Parameters
    ----------
    x: ndarray, OCLArray
        the input image
    res_g: OCLArray
        if given, us this as result buffer
        the dtype of res_g has to be compatible with the input dtype. 
        One can get the correct dtype via 
          dtype_out = gputools.transforms.integral_image._output_type_dict[dtype_input]
        If None, will be automatically created.
        
    tmp_g: OCLArray,
        temporary Array of same type as res_g. 
        If None, will be automatically created. 
        
    Returns
    -------
    res: ndarray, OCLArray
        the integral image as either ndarray or OCLArray (depending on the input)

    """
    if x.ndim == 2:
        if isinstance(x, OCLArray):
            return _integral2_buf(x, res_g=res_g, tmp_g=tmp_g)
        else:
            return _integral2_np(x)
    elif x.ndim == 3:
        if isinstance(x, OCLArray):
            return _integral3_buf(x, res_g=res_g, tmp_g=tmp_g)
        else:
            return _integral3_np(x)
    else:
        raise NotImplementedError("dim = %s not supported" % (x.ndim))


def _integral2_np(x):
    x_g = OCLArray.from_array(x)
    res_g = _integral2_buf(x_g)
    return res_g.get()



def _integral3_np(x):
    x_g = OCLArray.from_array(x)
    res_g = _integral3_buf(x_g)
    return res_g.get()


def _integral2_buf(x_g, res_g = None, tmp_g=None):
    if not x_g.dtype.type in _output_type_dict:
        raise ValueError("dtype %s currently not supported! (%s)" % (x_g.dtype.type, str(_output_type_dict.keys())))

    dtype_out = _output_type_dict[x_g.dtype.type]
    cl_dtype_in = cl_buffer_datatype_dict[x_g.dtype.type]
    cl_dtype_out = cl_buffer_datatype_dict[dtype_out]

    dtype_itemsize = np.dtype(dtype_out).itemsize

    max_local_size = get_device().get_info("MAX_WORK_GROUP_SIZE")
    prog = OCLProgram(abspath("kernels/integral_image.cl"),
                      build_options=["-D", "DTYPE=%s" % cl_dtype_out])

    if x_g.dtype.type != dtype_out:
        x_g = x_g.astype(dtype_out)

    if tmp_g is None:
        tmp_g = OCLArray.empty(x_g.shape, dtype_out)
    if res_g is None:
        res_g = OCLArray.empty(x_g.shape, dtype_out)

    ny, nx = x_g.shape

    def _scan_single(src, dst, ns, strides):
        nx, ny = ns
        stride_x, stride_y = strides
        loc = min(next_power_of_2(nx // 2), max_local_size // 2)
        nx_block = 2 * loc
        nx_pad = math.ceil(nx / nx_block) * nx_block

        nblocks = math.ceil(nx_pad // 2 / loc)
        sum_blocks = OCLArray.empty((ny, nblocks), dst.dtype)
        shared = cl.LocalMemory(2 * dtype_itemsize * loc)
        for b in range(nblocks):
            offset = b * loc
            prog.run_kernel("scan2d", (loc, ny), (loc, 1),
                            src.data, dst.data, sum_blocks.data, shared,
                            np.int32(nx_block), np.int32(stride_x), np.int32(stride_y), np.int32(offset), np.int32(b),
                            np.int32(nblocks), np.int32(nx))
        if nblocks > 1:
            _scan_single(sum_blocks, sum_blocks, (nblocks, ny), (1, nblocks))
            prog.run_kernel("add_sums2d", (nx_pad, ny), (nx_block, 1),
                            sum_blocks.data, dst.data,
                            np.int32(stride_x), np.int32(stride_y), np.int32(nblocks), np.int32(nx))

    _scan_single(x_g, tmp_g, (nx, ny), (1, nx))
    _scan_single(tmp_g, res_g, (ny, nx), (nx, 1))

    return res_g




def _integral3_buf(x_g, res_g = None, tmp_g = None):
    if not x_g.dtype.type in _output_type_dict:
        raise ValueError("dtype %s currently not supported! (%s)" % (x_g.dtype.type, str(_output_type_dict.keys())))

    dtype_out = _output_type_dict[x_g.dtype.type]
    cl_dtype_in = cl_buffer_datatype_dict[x_g.dtype.type]
    cl_dtype_out = cl_buffer_datatype_dict[dtype_out]

    dtype_itemsize = np.dtype(dtype_out).itemsize

    max_local_size = get_device().get_info("MAX_WORK_GROUP_SIZE")
    prog = OCLProgram(abspath("kernels/integral_image.cl"),
                      build_options=["-D", "DTYPE=%s" % cl_dtype_out])
    if x_g.dtype.type != dtype_out:
        x_g = x_g.astype(dtype_out)

    if tmp_g is None:
        tmp_g = OCLArray.empty(x_g.shape, dtype_out)
    if res_g is None:
        res_g = OCLArray.empty(x_g.shape, dtype_out)

    assert_bufs_type(dtype_out, tmp_g, res_g)

    nz, ny, nx = x_g.shape

    def _scan_single(src, dst, ns, strides):
        nx, ny, nz = ns
        stride_x, stride_y, stride_z = strides
        loc = min(next_power_of_2(nx // 2), max_local_size // 2)
        nx_block = 2 * loc
        nx_pad = math.ceil(nx / nx_block) * nx_block

        nblocks = math.ceil(nx_pad // 2 / loc)
        sum_blocks = OCLArray.empty((nz, ny, nblocks), dst.dtype)
        shared = cl.LocalMemory(2 * dtype_itemsize * loc)
        for b in range(nblocks):
            offset = b * loc
            prog.run_kernel("scan3d", (loc, ny, nz), (loc, 1, 1),
                            src.data, dst.data, sum_blocks.data, shared,
                            np.int32(nx_block),
                            np.int32(stride_x), np.int32(stride_y), np.int32(stride_z), np.int32(offset), np.int32(b),
                            np.int32(nblocks), np.int32(ny), np.int32(nx))
        if nblocks > 1:
            _scan_single(sum_blocks, sum_blocks, (nblocks, ny, nz), (1, nblocks, nblocks * ny))
            prog.run_kernel("add_sums3d", (nx_pad, ny, nz), (nx_block, 1, 1),
                            sum_blocks.data, dst.data,
                            np.int32(stride_x), np.int32(stride_y), np.int32(stride_z),
                            np.int32(nblocks), np.int32(ny), np.int32(nx))

    _scan_single(x_g, res_g, (nx, ny, nz), (1, nx, nx * ny))
    _scan_single(res_g, tmp_g, (ny, nx, nz), (nx, 1, nx * ny))
    _scan_single(tmp_g, res_g, (nz, nx, ny), (ny * nx, 1, nx))

    return res_g
