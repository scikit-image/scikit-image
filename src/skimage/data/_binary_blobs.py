import skimage2 as ski2

from .._shared._warnings import warn_external, PendingSkimage2Change


def binary_blobs(
    length=512,
    blob_size_fraction=0.1,
    n_dim=2,
    volume_fraction=0.5,
    rng=None,
    *,
    boundary_mode='nearest',
):
    """
    Generate synthetic binary image with several rounded blob-like objects.

    Parameters
    ----------
    length : int, optional
        Linear size of output image.
    blob_size_fraction : float, optional
        Typical linear size of blob, as a fraction of ``length``, should be
        smaller than 1.
    n_dim : int, optional
        Number of dimensions of output image.
    volume_fraction : float, default 0.5
        Fraction of image pixels covered by the blobs. Should be in [0, 1].
    rng : {`numpy.random.Generator`, int}, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).
        If `rng` is an int, it is used to seed the generator.
    boundary_mode : {'nearest', 'wrap'}, optional
        The blobs are created by smoothing and then thresholding an array
        consisting of ones at seed positions. This mode determines which
        values are filled in when the smoothing kernel overlaps the seed array's
        boundary.

        'nearest' (`a a a a | a b c d | d d d d`)
            By default, when applying the Gaussian filter, the seed array is
            extended by replicating the last boundary value. This will increase
            the size of blobs whose seed or center lies exactly on the edge.

        'wrap' (`a b c d | a b c d | a b c d`)
            The seed array is extended by wrapping around to the opposite edge.
            The resulting blob array can be tiled and blobs will be contiguous
            and have smooth edges across tile boundaries.

    Returns
    -------
    blobs : ndarray of bools
        Output binary image.

    Examples
    --------
    >>> from skimage import data
    >>> data.binary_blobs(length=5, blob_size_fraction=0.2)  # doctest: +SKIP
    array([[ True, False,  True,  True,  True],
           [ True,  True,  True, False,  True],
           [False,  True, False,  True,  True],
           [ True, False, False,  True,  True],
           [ True, False, False, False,  True]])
    >>> blobs = data.binary_blobs(length=256, blob_size_fraction=0.1)
    >>> # Finer structures
    >>> blobs = data.binary_blobs(length=256, blob_size_fraction=0.05)
    >>> # Blobs cover a smaller volume fraction of the image
    >>> blobs = data.binary_blobs(length=256, volume_fraction=0.3)
    """
    warn_external(
        "`skimage.data.binary_blobs` is deprecated in favor of "
        "`skimage2.data.binary_blobs` which has a new signature. "
        "Parameters `length` and `n_dim` have been replaced with `shape`. "
        "`blob_size_fraction` has been changed to `blob_size`. "
        "The default of `boundary_mode` has been changed to 'wrap'. "
        "To keep the old (`skimage`, v1.x) behavior, use:\n"
        "\n"
        "    import skimage2 as ski2\n"
        "    ski2.data.binary_blobs(\n"
        "        shape=(length,) * n_dim,\n"
        "        blob_size=blob_size_fraction * length,\n"
        "        boundary_mode='nearest',\n"
        "        ...\n"
        "    )",
        category=PendingSkimage2Change,
    )
    blob_size = blob_size_fraction * length
    return ski2.data.binary_blobs(
        shape=(length,) * n_dim,
        blob_size=blob_size,
        volume_fraction=volume_fraction,
        rng=rng,
        boundary_mode=boundary_mode,
    )
