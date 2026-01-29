"""Synthetic data generation."""

import numpy as np

from skimage._shared.filters import gaussian
from skimage._shared._warnings import warn_external


def binary_blobs(
    shape,
    *,
    blob_size,
    volume_fraction=0.5,
    rng=None,
    boundary_mode='wrap',
):
    """Generate synthetic binary image containing blob-like objects.

    Parameters
    ----------
    shape : tuple of (int, ...)
        Shape of the output image.
    blob_size : float
        Typical linear size of blob in pixels.
        Values smaller than 1 may lead to unexpected results.
    volume_fraction : float, default 0.5
        Fraction of image pixels covered by the blobs. Higher value lead to
        a larger fraction of pixels being part of blobs. Should be in [0, 1].
    rng : int or :class:`numpy.random.Generator`, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).
        If `rng` is an int, it is used to seed the generator.
    boundary_mode : {'wrap', 'nearest'}, optional
        The blobs are created by smoothing and then thresholding an
        array consisting of ones at seed positions. This mode determines which
        values are filled in when the smoothing kernel overlaps the seed
        array's boundary.

        'wrap' (`a b c d | a b c d | a b c d`)
            By default, the seed array is extended by wrapping around to the
            opposite edge. The resulting blob array can be tiled and blobs will
            be contiguous and have smooth edges across tile boundaries.

        'nearest' (`a a a a | a b c d | d d d d`)
            When applying the Gaussian filter, the seed array is extended by
            replicating the last boundary value. This will increase the size of
            blobs whose seed or center lies exactly on the edge.

    Returns
    -------
    blobs : ndarray of dtype bool
        Output binary image.

    Examples
    --------
    >>> import skimage2 as sk2
    >>> sk2.data.binary_blobs(shape=(5, 5), blob_size=1)  # doctest: +SKIP
    array([[ True, False,  True,  True,  True],
           [ True,  True,  True, False,  True],
           [False,  True, False,  True,  True],
           [ True, False, False,  True,  True],
           [ True, False, False, False,  True]])
    >>> blobs = sk2.data.binary_blobs(shape=(256, 256), blob_size=25)
    >>> # Finer structures
    >>> blobs = sk2.data.binary_blobs(shape=(256, 256), blob_size=13)
    >>> # Blobs cover a smaller volume fraction of the image
    >>> blobs = sk2.data.binary_blobs(
    ...     shape=(256, 256), blob_size=25, volume_fraction=0.3
    ... )
    """
    if boundary_mode not in {"nearest", "wrap"}:
        raise ValueError(f"unsupported `boundary_mode`: {boundary_mode!r}")

    min_length = min(shape)
    blob_size_fraction = blob_size / min_length

    if blob_size < 1:
        warn_external(
            f"Requested `blob_size` ({blob_size}) is smaller than 1. "
            f"Small blob sizes may lead to unexpected results!",
            category=RuntimeWarning,
        )
    if blob_size < 0.1:
        blob_size_fraction = 0.1 / min_length
        warn_external(
            "Clamping to `blob_size=0.1` to avoid allocating excessive memory.",
            category=RuntimeWarning,
        )

    rng = np.random.default_rng(rng)
    mask = np.zeros(shape)

    n_dim = len(shape)
    n_pts = max(int(1.0 / blob_size_fraction) ** n_dim, 1)

    points = rng.random((n_dim, n_pts))
    for ax, length in enumerate(shape):
        points[ax] *= length
    points = points.astype(int)

    mask[tuple(points)] = 1
    mask = gaussian(
        mask,
        sigma=0.25 * min_length * blob_size_fraction,
        preserve_range=False,
        mode=boundary_mode,
    )
    threshold = np.quantile(mask, 1 - volume_fraction)
    blobs = mask >= threshold
    return blobs
