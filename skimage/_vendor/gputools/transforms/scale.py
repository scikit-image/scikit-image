""" some image manipulation functions like scaling, rotating, etc...

"""

from __future__ import print_function, unicode_literals, absolute_import, division
import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
from gputools import OCLArray, OCLImage, OCLProgram, get_device
from gputools import OCLElementwiseKernel

from ._abspath import abspath


def _scale_shape(dshape, scale = (1,1,1)):
    """returns the shape after scaling (should be the same as ndimage.zoom"""
    nshape = np.round(np.array(dshape) * np.array(scale))
    return tuple(nshape.astype(np.int))


def scale(data, scale = (1.,1.,1.), interpolation = "linear"):
    """
    returns a interpolated, scaled version of data
    
    the output shape is scaled too.
    
    Parameters
    ----------
    data: ndarray
        3d input array
    scale: float, tuple
        scaling factor along each axis (x,y,z) 
    interpolation: str
        either "nearest" or "linear"

    Returns
    -------
        scaled output 

    """


    if not (isinstance(data, np.ndarray) and data.ndim == 3):
        raise ValueError("input data has to be a 3d array!")

    interpolation_defines = {"linear": ["-D", "SAMPLER_FILTER=CLK_FILTER_LINEAR"],
                             "nearest": ["-D", "SAMPLER_FILTER=CLK_FILTER_NEAREST"]}

    if not interpolation in interpolation_defines:
        raise KeyError(
            "interpolation = '%s' not defined ,valid: %s" % (interpolation, list(interpolation_defines.keys())))


    options_types = {np.uint8:["-D","TYPENAME=uchar","-D","READ_IMAGE=read_imageui"],
                    np.uint16: ["-D","TYPENAME=short","-D", "READ_IMAGE=read_imageui"],
                    np.float32: ["-D","TYPENAME=float", "-D","READ_IMAGE=read_imagef"],
                    }

    dtype = data.dtype.type

    if not dtype in options_types:
        raise ValueError("type %s not supported! Available: %s"%(dtype ,str(list(options_types.keys()))))


    if not isinstance(scale,(tuple, list, np.ndarray)):
        scale = (scale,)*3

    if len(scale) != 3:
        raise ValueError("scale = %s misformed"%scale)

    d_im = OCLImage.from_array(data)

    nshape = _scale_shape(data.shape,scale)

    res_g = OCLArray.empty(nshape,dtype)


    prog = OCLProgram(abspath("kernels/scale.cl"),
                      build_options=interpolation_defines[interpolation]+options_types[dtype ])


    prog.run_kernel("scale",
                    res_g.shape[::-1],None,
                    d_im,res_g.data)

    return res_g.get()

#
# def scale_bicubic(data, scale=(1., 1., 1.)):
#     """
#     returns a interpolated, scaled version of data
#
#     the output shape is scaled too.
#
#     Parameters
#     ----------
#     data: ndarray
#         3d input array
#     scale: float, tuple
#         scaling factor along each axis (x,y,z)
#     interpolation: str
#         either "nearest" or "linear"
#
#     Returns
#     -------
#         scaled output
#
#     """
#
#     if not (isinstance(data, np.ndarray) and data.ndim == 3):
#         raise ValueError("input data has to be a 3d array!")
#
#     options_types = {np.uint8: ["-D", "TYPENAME=uchar", "-D", "READ_IMAGE=read_imageui"],
#                      np.uint16: ["-D", "TYPENAME=short", "-D", "READ_IMAGE=read_imageui"],
#                      np.float32: ["-D", "TYPENAME=float", "-D", "READ_IMAGE=read_imagef"],
#                      }
#
#     dtype = data.dtype.type
#
#     if not dtype in options_types:
#         raise ValueError("type %s not supported! Available: %s" % (dtype, str(list(options_types.keys()))))
#
#     if not isinstance(scale, (tuple, list, np.ndarray)):
#         scale = (scale,) * 3
#
#     if len(scale) != 3:
#         raise ValueError("scale = %s misformed" % scale)
#
#     d_im = OCLImage.from_array(data)
#
#     nshape = _scale_shape(data.shape, scale)
#
#     res_g = OCLArray.empty(nshape, dtype)
#
#     prog = OCLProgram(abspath("kernels/scale.cl"),
#                       build_options=options_types[dtype])
#
#     prog.run_kernel("scale_bicubic",
#                     res_g.shape[::-1], None,
#                     d_im, res_g.data)
#
#     return res_g.get()

