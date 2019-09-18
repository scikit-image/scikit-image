""" scaling images

"""

from __future__ import print_function, unicode_literals, absolute_import, division
import logging

logger = logging.getLogger(__name__)

import os
import numpy as np
import warnings
from gputools import OCLArray, OCLImage, OCLProgram
from gputools.core.ocltypes import cl_buffer_datatype_dict
from gputools.utils import mat4_rotate, mat4_translate
from ._abspath import abspath
from mako.template import Template


def affine(data, mat=np.identity(4), mode="constant", interpolation="linear"):
    """
    affine transform data with matrix mat, which is the inverse coordinate transform matrix  
    (similar to ndimage.affine_transform)
     
    Parameters
    ----------
    data, ndarray
        3d array to be transformed
    mat, ndarray 
        3x3 or 4x4 inverse coordinate transform matrix 
    mode: string 
        boundary mode, one of the following:
        'constant'
            pads with zeros 
        'edge'
            pads with edge values
        'wrap'
            pads with the repeated version of the input 
    interpolation, string
        interpolation mode, one of the following    
        'linear'
        'nearest'
        
    Returns
    -------
    res: ndarray
        transformed array (same shape as input)
        
    """
    warnings.warn(
        "gputools.transform.affine: API change as of gputools>= 0.2.8: the inverse of the matrix is now used as in scipy.ndimage.affine_transform")

    if not (isinstance(data, np.ndarray) and data.ndim == 3):
        raise ValueError("input data has to be a 3d array!")

    interpolation_defines = {"linear": ["-D", "SAMPLER_FILTER=CLK_FILTER_LINEAR"],
                             "nearest": ["-D", "SAMPLER_FILTER=CLK_FILTER_NEAREST"]}

    mode_defines = {"constant": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_CLAMP"],
                    "wrap": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_REPEAT"],
                    "edge": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_CLAMP_TO_EDGE"]
                    }

    if not interpolation in interpolation_defines:
        raise KeyError(
            "interpolation = '%s' not defined ,valid: %s" % (interpolation, list(interpolation_defines.keys())))

    if not mode in mode_defines:
        raise KeyError("mode = '%s' not defined ,valid: %s" % (mode, list(mode_defines.keys())))

    # reorder matrix, such that x,y,z -> z,y,x (as the kernel is assuming that)

    d_im = OCLImage.from_array(data.astype(np.float32, copy=False))
    res_g = OCLArray.empty(data.shape, np.float32)
    mat_inv_g = OCLArray.from_array(mat.astype(np.float32, copy=False))

    prog = OCLProgram(abspath("kernels/affine.cl")
                      , build_options=interpolation_defines[interpolation] +
                                      mode_defines[mode])

    prog.run_kernel("affine3",
                    data.shape[::-1], None,
                    d_im, res_g.data, mat_inv_g.data)

    return res_g.get()


def shift(data, shift=(0, 0, 0), mode="constant", interpolation="linear"):
    """
    translates 3d data by given amount
  
    
    Parameters
    ----------
    data: ndarray
        3d array
    shift : float or sequence
        The shift along the axes. If a float, `shift` is the same for each axis. 
        If a sequence, `shift` should contain one value for each axis.    
    mode: string 
        boundary mode, one of the following:      
        'constant'
            pads with zeros 
        'edge'
            pads with edge values
        'wrap'
            pads with the repeated version of the input 
    interpolation, string
        interpolation mode, one of the following       
        'linear'
        'nearest'
        
    Returns
    -------
    res: ndarray
        shifted array (same shape as input)
    """
    if np.isscalar(shift):
        shift = (shift,) * 3

    if len(shift) != 3:
        raise ValueError("shift (%s) should be of length 3!")

    shift = -np.array(shift)
    return affine(data, mat4_translate(*shift), mode=mode, interpolation=interpolation)


def rotate(data, axis=(1., 0, 0), angle=0., center=None, mode="constant", interpolation="linear"):
    """
    rotates data around axis by a given angle

    Parameters
    ----------
    data: ndarray
        3d array
    axis: tuple
        axis to rotate by angle about
        axis = (x,y,z)
    angle: float
    center: tuple or None
        origin of rotation (cz,cy,cx) in pixels
        if None, center is the middle of data
    
    mode: string 
        boundary mode, one of the following:        
        'constant'
            pads with zeros 
        'edge'
            pads with edge values
        'wrap'
            pads with the repeated version of the input 
    interpolation, string
        interpolation mode, one of the following      
        'linear'
        'nearest'
        
    Returns
    -------
    res: ndarray
        rotated array (same shape as input)

    """
    if center is None:
        center = tuple([s // 2 for s in data.shape])

    cx, cy, cz = center
    m = np.dot(mat4_translate(cx, cy, cz),
               np.dot(mat4_rotate(angle, *axis),
                      mat4_translate(-cx, -cy, -cz)))
    m = np.linalg.inv(m)
    return affine(data, m, mode=mode, interpolation=interpolation)


def map_coordinates(data, coordinates, interpolation="linear",
                    mode='constant'):
    """
    Map data to new coordinates by interpolation.
    The array of coordinates is used to find, for each point in the output,
    the corresponding coordinates in the input.

    should correspond to scipy.ndimage.map_coordinates
    
    Parameters
    ----------
    data
    coordinates
    output
    interpolation
    mode
    cval
    prefilter

    Returns
    -------
    """
    if not (isinstance(data, np.ndarray) and data.ndim in (2, 3)):
        raise ValueError("input data has to be a 2d or 3d array!")

    coordinates = np.asarray(coordinates, np.int32)
    if not (coordinates.shape[0] == data.ndim):
        raise ValueError("coordinate has to be of shape (data.ndim,m) ")

    interpolation_defines = {"linear": ["-D", "SAMPLER_FILTER=CLK_FILTER_LINEAR"],
                             "nearest": ["-D", "SAMPLER_FILTER=CLK_FILTER_NEAREST"]}

    mode_defines = {"constant": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_CLAMP"],
                    "wrap": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_REPEAT"],
                    "edge": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_CLAMP_TO_EDGE"]
                    }

    if not interpolation in interpolation_defines:
        raise KeyError(
            "interpolation = '%s' not defined ,valid: %s" % (interpolation, list(interpolation_defines.keys())))

    if not mode in mode_defines:
        raise KeyError("mode = '%s' not defined ,valid: %s" % (mode, list(mode_defines.keys())))

    if not data.dtype.type in cl_buffer_datatype_dict:
        raise KeyError("dtype %s not supported yet (%s)" % (data.dtype.type, tuple(cl_buffer_datatype_dict.keys())))

    dtype_defines = ["-D", "DTYPE=%s" % cl_buffer_datatype_dict[data.dtype.type]]

    d_im = OCLImage.from_array(data)
    coordinates_g = OCLArray.from_array(coordinates.astype(np.float32, copy=False))
    res_g = OCLArray.empty(coordinates.shape[1], data.dtype)

    prog = OCLProgram(abspath("kernels/map_coordinates.cl")
                      , build_options=interpolation_defines[interpolation] +
                                      mode_defines[mode] + dtype_defines)

    kernel = "map_coordinates{ndim}".format(ndim=data.ndim)

    prog.run_kernel(kernel,
                    (coordinates.shape[-1],), None,
                    d_im, res_g.data, coordinates_g.data)

    return res_g.get()


def geometric_transform(data, mapping = "c0,c1", output_shape=None,
                        mode='constant', interpolation="linear"):
    """
    Apply an arbitrary geometric transform.
    The given mapping function is used to find, for each point in the
    output, the corresponding coordinates in the input. The value of the
    input at those coordinates is determined by spline interpolation of
    the requested order.
    Parameters
    ----------
    %(input)s
    mapping : {callable, scipy.LowLevelCallable}
        A callable object that accepts a tuple of length equal to the output
        array rank, and returns the corresponding input coordinates as a tuple
        of length equal to the input array rank.
    """

    if not (isinstance(data, np.ndarray) and data.ndim in (2, 3)):
        raise ValueError("input data has to be a 2d or 3d array!")

    interpolation_defines = {"linear": ["-D", "SAMPLER_FILTER=CLK_FILTER_LINEAR"],
                             "nearest": ["-D", "SAMPLER_FILTER=CLK_FILTER_NEAREST"]}

    mode_defines = {"constant": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_CLAMP"],
                    "wrap": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_REPEAT"],
                    "edge": ["-D", "SAMPLER_ADDRESS=CLK_ADDRESS_CLAMP_TO_EDGE"]
                    }

    if not interpolation in interpolation_defines:
        raise KeyError(
            "interpolation = '%s' not defined ,valid: %s" % (interpolation, list(interpolation_defines.keys())))

    if not mode in mode_defines:
        raise KeyError("mode = '%s' not defined ,valid: %s" % (mode, list(mode_defines.keys())))

    if not data.dtype.type in cl_buffer_datatype_dict:
        raise KeyError("dtype %s not supported yet (%s)" % (data.dtype.type, tuple(cl_buffer_datatype_dict.keys())))

    dtype_defines = ["-D", "DTYPE={type}".format(type=cl_buffer_datatype_dict[data.dtype.type])]

    image_functions = {np.float32:"read_imagef",
                       np.uint8: "read_imageui",
                       np.uint16: "read_imageui",
                       np.int32: "read_imagei"}

    image_read_defines = ["-D","READ_IMAGE=%s"%image_functions[data.dtype.type]]

    with open(abspath("kernels/geometric_transform.cl"), "r") as f:
        tpl = Template(f.read())

    output_shape = tuple(output_shape)

    mappings = {"FUNC2": "c1,c0",
                "FUNC3": "c2,c1,c0"}

    mappings["FUNC%d" % data.ndim] = ",".join(reversed(mapping.split(",")))

    rendered = tpl.render(**mappings)

    d_im = OCLImage.from_array(data)
    res_g = OCLArray.empty(output_shape, data.dtype)

    prog = OCLProgram(src_str=rendered,
                      build_options=interpolation_defines[interpolation] +
                                    mode_defines[mode] + dtype_defines+image_read_defines)

    kernel = "geometric_transform{ndim}".format(ndim=data.ndim)

    prog.run_kernel(kernel,
                    output_shape[::-1], None,
                    d_im, res_g.data)

    return res_g.get()


if __name__ == '__main__':
    d = np.zeros((200, 200, 200), np.float32)
    d[20:-20, 20:-20, 20:-20] = 1.

    # res = translate(d, x = 10, y = 5, z= -10 )
    res = rotate(d, center=(100, 100, 100), angle=.5)
