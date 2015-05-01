import numpy as np
from ..filters import gaussian_filter


def binary_blobs(length=512, blob_size_fraction=0.1, n_dim=2,
                 volume_fraction=0.5, seed=None):
    """
    Generate synthetic binary image with several blob-like rounded objects.

    Parameters
    ----------
    length : int, default 512
        Linear size of output image.
    blob_size_fraction : float, default 0.1
        Typical linear size of blob, as a fraction of ``length``, should be
        smaller than 1.
    n_dim : int, default 2
        Number of dimensions of output image.
    volume_fraction : float, default 0.5
        Fraction of image pixels covered by the blobs (where the output is 1).
        Should be in [0, 1].
    seed : int, default 0
        Seed to initialize the random number generator.

    Returns
    -------
    blobs : ndarray of bools
        Output binary image

    Examples
    --------
    >>> blobs = binary_blobs(length=256, blob_size_fraction=0.1)
    >>> # Finer structures
    >>> blobs = binary_blobs(length=256, blob_size_fraction=0.05)
    >>> # Blobs cover a smaller volume fraction of the image
    >>> blobs = binary_blobs(length=256, volume_fraction=0.3)
    """
    if seed is None:
        seed = 0
    # Fix the seed for reproducible results
    rs = np.random.RandomState(seed)
    shape = tuple([length] * n_dim)
    mask = np.zeros(shape)
    n_pts = max(int(1. / blob_size_fraction) ** n_dim, 1)
    points = (length * rs.rand(n_dim, n_pts)).astype(np.int)
    mask[[indices for indices in points]] = 1
    mask = gaussian_filter(mask, sigma=0.25 * length * blob_size_fraction)
    threshold = np.percentile(mask, 100 * (1 - volume_fraction))
    return np.logical_not(mask < threshold)
