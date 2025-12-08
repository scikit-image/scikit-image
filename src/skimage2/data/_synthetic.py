"""Synthetic data generation."""

from skimage.data._binary_blobs import _binary_blobs_sk2_implementation


def binary_blobs(
    shape,
    *,
    blob_size_fraction=0.1,
    volume_fraction=0.5,
    rng=None,
    boundary_mode='wrap',
):
    """Generate synthetic binary image with several rounded blob-like objects.

    Parameters
    ----------
    shape : tuple of int(s)
        Shape of the output image.
    blob_size_fraction : float, optional
        Typical linear size of blob, as a fraction of the length of the shortest
        dimension in `shape`. Should be smaller than 1.
    volume_fraction : float, default 0.5
        Fraction of image pixels covered by the blobs (where the output is 1).
        Should be in [0, 1].
    rng : {`numpy.random.Generator`, int}, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).
        If `rng` is an int, it is used to seed the generator.
    boundary_mode : {'wrap', 'nearest'}, optional
        The blobs are created by smoothing and then thresholding an
        array consisting of ones at seed positions. This mode determines which
        values are  filled in when the smoothing kernel overlaps the seed
        array's boundary.

        'wrap' (`a b c d | a b c d | a b c d`)
            By default, the seed array is extended by wrapping around to the
            opposite edge. The resulting blob array can be tiled and blobs will
            be contiguous and  have smooth edges across tile boundaries.

        'nearest' (`a a a a | a b c d | d d d d`)
            When applying the Gaussian filter, the seed array is extended by
            replicating the last boundary value. This will increase the size of
            blobs whose seed or center lies exactly on the edge.

    boundary_mode : str, default "nearest"
        The `mode` parameter passed to the Gaussian filter.
        Use "wrap" for periodic boundary conditions.

    Returns
    -------
    blobs : ndarray of bools
        Output binary image

    Examples
    --------
    >>> import skimage2 as ski
    >>> ski.data.binary_blobs(shape=(5, 5), blob_size_fraction=0.2)  # doctest: +SKIP
    array([[ True, False,  True,  True,  True],
           [ True,  True,  True, False,  True],
           [False,  True, False,  True,  True],
           [ True, False, False,  True,  True],
           [ True, False, False, False,  True]])
    >>> blobs = ski.data.binary_blobs(shape=(256, 256), blob_size_fraction=0.1)
    >>> # Finer structures
    >>> blobs = ski.data.binary_blobs(shape=(256, 256), blob_size_fraction=0.05)
    >>> # Blobs cover a smaller volume fraction of the image
    >>> blobs = ski.data.binary_blobs(shape=(256, 256), volume_fraction=0.3)
    """
    return _binary_blobs_sk2_implementation(
        shape=shape,
        blob_size_fraction=blob_size_fraction,
        volume_fraction=volume_fraction,
        rng=rng,
        boundary_mode=boundary_mode,
    )
