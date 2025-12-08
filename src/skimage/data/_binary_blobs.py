import warnings

import numpy as np

from .._shared.filters import gaussian


def _binary_blobs_sk2_implementation(
    shape,
    *,
    blob_size_fraction,
    volume_fraction,
    rng,
    boundary_mode,
):
    """Generate synthetic binary image with several rounded blob-like objects.

    Revised implementation for skimage2.
    See full docstring at :func:`skimage2.data.binary_blobs`.
    """
    if boundary_mode not in {"nearest", "wrap"}:
        raise ValueError(f"unsupported `boundary_mode`: {boundary_mode!r}")

    min_length = min(shape)
    blob_size = blob_size_fraction * min_length
    if blob_size < 0.1:
        clamped_size_fraction = 0.1 / min_length
        clamped_blob_size = clamped_size_fraction * min_length
        warnings.warn(
            f"`{blob_size_fraction=}` together with `{shape=}` would result in a blob "
            f"size of {blob_size} pixels. Small blob sizes likely lead to unexpected "
            f"results! "
            f"Clamping to `blob_size_fraction={clamped_size_fraction}` and a blob size "
            f"of {clamped_blob_size} pixels to avoid allocating excessive memory.",
            category=RuntimeWarning,
            stacklevel=3,
        )
        blob_size_fraction = clamped_size_fraction

    rng = np.random.default_rng(rng)
    mask = np.zeros(shape)

    n_dim = len(shape)
    n_pts = max(int(1.0 / blob_size_fraction) ** n_dim, 1)

    points = rng.random((n_dim, n_pts))
    for ax, length in enumerate(shape):
        points[ax] *= length
    points = points.astype(int)

    mask[tuple(indices for indices in points)] = 1
    mask = gaussian(
        mask,
        sigma=0.25 * min_length * blob_size_fraction,
        preserve_range=False,
        mode=boundary_mode,
    )
    threshold = np.percentile(mask, 100 * (1 - volume_fraction))
    return np.logical_not(mask < threshold)


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
        Fraction of image pixels covered by the blobs (where the output is 1).
        Should be in [0, 1].
    rng : {`numpy.random.Generator`, int}, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).
        If `rng` is an int, it is used to seed the generator.
    boundary_mode : {'nearest', 'wrap'}, optional
        The blobs are created by smoothing and then thresholding an
        array consisting of ones at seed positions. This mode determines which values are
        filled in when the smoothing kernel overlaps the seed array's boundary.

        'nearest' (`a a a a | a b c d | d d d d`)
            By default, when applying the Gaussian filter, the seed array is extended by replicating the last
            boundary value. This will increase the size of blobs whose seed or
            center lies exactly on the edge.

        'wrap' (`a b c d | a b c d | a b c d`)
            The seed array is extended by wrapping around to the opposite edge.
            The resulting blob array can be tiled and blobs will be contiguous and
            have smooth edges across tile boundaries.

    boundary_mode : str, default "nearest"
        The `mode` parameter passed to the Gaussian filter.
        Use "wrap" for periodic boundary conditions.

    Returns
    -------
    blobs : ndarray of bools
        Output binary image

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
    return _binary_blobs_sk2_implementation(
        shape=(length,) * n_dim,
        blob_size_fraction=blob_size_fraction,
        volume_fraction=volume_fraction,
        rng=rng,
        boundary_mode=boundary_mode,
    )
