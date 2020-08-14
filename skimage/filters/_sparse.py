import numpy as np
from ._sparse_cy import _correlate_sparse_offsets

def correlate_sparse(image, kernel, mode='reflect'):
    """Compute valid cross-correlation of `padded_array` and `kernel`.

    This function is *fast* when `kernel` is large with many zeros.

    See ``scipy.ndimage.correlate`` for a description of cross-correlation.

    Parameters
    ----------
    image : array of float, shape (M, N,[ ...,] P)
        The input array. If mode is 'valid', this array should already be
        padded, as a margin of the same shape as kernel will be stripped
        off.
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
        padded_image = np.pad(
            image,
            [(w//2, w//2) for w in kernel.shape],
            mode=mode,
        )
    indices = np.nonzero(kernel)
    offsets = np.ravel_multi_index(indices, padded_image.shape).astype(np.intp)
    values = kernel[indices].astype(padded_image.dtype)

    result = np.zeros(
        [a - b + 1 for a, b in zip(padded_image.shape, kernel.shape)],
        dtype=padded_image.dtype,
    )

    # memory-efficient version of numpy.mgrid
    corner_multi_indices = np.broadcast_arrays(
                        *np.meshgrid(
                            *[np.arange(i) for i in result.shape], 
                            indexing='ij', 
                            sparse=True
                            )
                        )

    corner_indices = np.ravel_multi_index(
        corner_multi_indices, padded_image.shape
    ).ravel().astype(np.intp)

    _correlate_sparse_offsets(padded_image.ravel(), corner_indices,
                              offsets, values, result.ravel())
    return result
