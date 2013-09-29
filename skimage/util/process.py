__all__ = ['process_blocks']

import numpy as np
from skimage.util import view_as_windows
from multiprocessing import Pool
from functools import partial


def process_blocks(image, block_shape, func, func_args={},
                   overlap=0, n_jobs=1):
    """Apply a function to distinct or overlapping blocks in the image.

    Parameters
    ----------
    image : ndarray
        Input image.
    block_shape : tuple
        Block size.
    func : callable, f(
        Function to be applied to each window.
    func_args : dict
        Additional arguments for `func`.
    overlap : int
        The amount of overlap between blocks.
    jobs : int
        The number of jobs to launch in parallel.

    Returns
    -------
    output : ndarray
        Outputs generated by applying the function to each block.

    Examples
    --------
    >>> from skimage.data import camera
    >>> image = camera()
    >>> output = process_windows(image, (8, 8), np.sum)
    >>>
    >>> from skimage.color import gray2rgb
    >>> output2 = process_windows(gray2rgb(image), (8, 8, 3),
    ...                           np.sum, {'axis': -1})

    """
    block_shape = np.asarray(block_shape)
    step = max(block_shape) - overlap

    if block_shape.size != image.ndim:
        raise ValueError("Block shape must correspond to image dimensions")

    image_views = view_as_windows(image, block_shape, step)
    out_shape = image_views.shape[:-block_shape.size]
    indicies = np.ndindex(*out_shape)

    if n_jobs > 1:
        func_map = partial(func, **func_args)
        pool = Pool(processes=n_jobs)
        output = np.array(pool.map(func_map,
                                   [image_views[idx] for idx in indicies]))
    else:
        output = np.array([func(image_views[idx], **func_args)
                           for idx in indicies])

    return output.reshape(out_shape + output.shape[1:])

