import numpy as np
from ._sparse_cy import _correlate_sparse_offsets


def broadcast_mgrid(arrays):
    shape = tuple(map(len, arrays))
    ndim = len(shape)
    result = []
    for i, arr in enumerate(arrays, start=1):
        reshaped = np.broadcast_to(arr[(...,) + (np.newaxis,) * (ndim - i)],
                                   shape)
        result.append(reshaped)
    return result


def correlate_sparse(image, kernel, mode='reflect'):
    """Compute valid cross-correlation of `padded_array` and `kernel`.

    This function is *fast* when `kernel` is large with many zeros.

    See ``scipy.ndimage.correlate`` for a description of cross-correlation.

    Parameters
    ----------
    image : array of float, shape (M, N,[ ...,] P)
        The input array. It should be already padded, as a margin of the
        same shape as kernel (-1) will be stripped off.
    kernel : array of float, shape (Q, R,[ ...,] S)
        The kernel to be correlated. Must have the same number of
        dimensions as `padded_array`. For high performance, it should
        be sparse (few nonzero entries).
    mode : string, optional
        See `np.pad` for valid modes. Additionally, mode 'valid' is
        accepted, in which case no padding is applied and the result is
        the result for the smaller image for which the kernel is entirely
        inside the original data.

    Returns
    -------
    result : array of float, shape (M, N,[ ...,] P)
        The result of cross-correlating `image` with `kernel`. If mode
        'valid' is used, the resulting shape is (M-Q+1, N-R+1,[ ...,] P-S+1).
    """
    if mode == 'valid':
        padded_image = image
    else:
        w = kernel.shape[0] // 2
        padded_image = np.pad(image, (w, w-1), mode=mode)
    indices = np.nonzero(kernel)
    offsets = np.ravel_multi_index(indices, padded_image.shape)
    values = kernel[indices]
    
    result = np.zeros([a - b + 1
                       for a, b in zip(padded_image.shape, kernel.shape)])
    corner_multi_indices = broadcast_mgrid([np.arange(i)
                                            for i in result.shape])
    corner_indices = np.ravel_multi_index(corner_multi_indices,
                                          padded_image.shape).ravel()

    _correlate_sparse_offsets(padded_image.ravel(), corner_indices,
                              offsets, values, result.ravel())
    return result