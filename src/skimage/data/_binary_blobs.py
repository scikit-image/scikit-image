import numpy as np

from .._shared.filters import gaussian


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
        intermediate array with seeds. This mode determines which values are
        filled in when the smoothing kernel overlaps the seed array's boundary.

        'nearest' (`a a a a | a b c d | d d d d`)
            By default, the seed array is extended by replicating the last
            boundary value. This will increase the size of blobs whose seed or
            center lies exactly on the edge.

        'wrap' (`a b c d | a b c d | a b c d`)
            The seed array is extended by wrapping around to the opposite edge.
            With this, the resulting array can be tiled (an edge is
            concatenated to its opposing edge) and blobs will be contigious and
            have smooth edges accross each tiles boundary.

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
    if boundary_mode not in {"nearest", "wrap"}:
        raise ValueError(f"unsupported `boundary_mode`: {boundary_mode!r}")

    rs = np.random.default_rng(rng)
    shape = tuple([length] * n_dim)
    mask = np.zeros(shape)
    n_pts = max(int(1.0 / blob_size_fraction) ** n_dim, 1)
    points = (length * rs.random((n_dim, n_pts))).astype(int)
    mask[tuple(indices for indices in points)] = 1
    mask = gaussian(
        mask,
        sigma=0.25 * length * blob_size_fraction,
        preserve_range=False,
        mode=boundary_mode,
    )
    threshold = np.percentile(mask, 100 * (1 - volume_fraction))
    return np.logical_not(mask < threshold)
