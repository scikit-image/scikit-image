# coding: utf-8
"""Common tools to optical flow algorithms.

"""

import numpy as np
import skimage
from skimage.transform import pyramid_reduce, resize


def resize_flow(flow, shape):
    """Rescale the values of the vector field (u, v) to the desired shape.

    The values of the output vector field are scaled to the new
    resolution.

    Parameters
    ----------
    flow : ~numpy.ndarray
        The motion field to be processed.
    shape : iterable
        Couple of integers representing the output shape.

    Returns
    -------
    rflow : ~numpy.ndarray
        The resized and rescaled motion field.

    """

    scale = np.array([n / o for n, o in zip(shape, flow.shape[1:])])

    for _ in shape:
        scale = scale[..., np.newaxis]

    return scale*resize(flow, (flow.shape[0],) + shape, order=0,
                        preserve_range=True, anti_aliasing=False)


def get_pyramid(I, downscale=2.0, nlevel=10, min_size=16):
    """Construct image pyramid.

    Parameters
    ----------
    I : ~numpy.ndarray
        The image to be preprocessed (Gray scale or RGB).
    downscale : float
        The pyramid downscale factor.
    nlevel : int
        The maximum number of pyramid levels.
    min_size : int
        The minimum size for any dimension of the pyramid levels.

    Returns
    -------
    pyramid : list[~numpy.ndarray]
        The coarse to fine images pyramid.

    """

    pyramid = [I]
    size = min(I.shape)
    count = 1

    while (count < nlevel) and (size > downscale * min_size):
        J = pyramid_reduce(pyramid[-1], downscale, multichannel=False)
        pyramid.append(J)
        size = min(J.shape)
        count += 1

    return pyramid[::-1]


def coarse_to_fine(I0, I1, solver, downscale=2, nlevel=10, min_size=16):
    """Generic coarse to fine solver.

    Parameters
    ----------
    I0 : ~numpy.ndarray
        The first gray scale image of the sequence.
    I1 : ~numpy.ndarray
        The second gray scale image of the sequence.
    solver : callable
        The solver applyed at each pyramid level.
    downscale : float
        The pyramid downscale factor.
    nlevel : int
        The maximum number of pyramid levels.
    min_size : int
        The minimum size for any dimension of the pyramid levels.

    Returns
    -------
    flow : ~numpy.ndarray
        The estimated optical flow.

    """

    if I0.shape != I1.shape:
        raise ValueError("Input images should have the same shape")

    pyramid = list(zip(get_pyramid(skimage.img_as_float32(I0),
                                   downscale, nlevel, min_size),
                       get_pyramid(skimage.img_as_float32(I1),
                                   downscale, nlevel, min_size)))

    flow = np.zeros((pyramid[0][0].ndim, ) + pyramid[0][0].shape)

    flow = solver(pyramid[0][0], pyramid[0][1], flow)

    for J0, J1 in pyramid[1:]:
        flow = solver(J0, J1, resize_flow(flow, J0.shape))

    return flow
