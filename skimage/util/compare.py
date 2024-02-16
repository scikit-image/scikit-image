import numpy as np

from .dtype import img_as_float
from itertools import product

from skimage._shared.utils import (
    deprecate_parameter,
    DEPRECATED,
)


@deprecate_parameter(
    "image2", new_name="image1", start_version="0.23", stop_version="0.25"
)
def compare_images(image0, image1, image2=DEPRECATED, *, method='diff', n_tiles=(8, 8)):
    """
    Return an image showing the differences between two images.

    .. versionadded:: 0.16

    Parameters
    ----------
    image0 : ndarray, shape (M, N)
        First input image.

        .. versionadded:: 0.23
    image1 : ndarray, shape (M, N)
        Second input image. Must be of the same shape as `image0`.

        .. versionchanged:: 0.23
            `image1` changed from being the name of the first image to that of
            the second image.
    method : string, optional
        Method used for the comparison.
        Valid values are {'diff', 'blend', 'checkerboard'}.
        Details are provided in the note section.

    .. versionchanged:: 0.23
            This parameter is now keyword-only.
    n_tiles : tuple, optional
        Used only for the `checkerboard` method. Specifies the number
        of tiles (row, column) to divide the image.

    Other Parameters
    ----------------
    image2 : DEPRECATED
        Deprecated in favor of `image1`.

        .. deprecated:: 0.23

    Returns
    -------
    comparison : ndarray, shape (M, N)
        Image showing the differences.

    Notes
    -----
    ``'diff'`` computes the absolute difference between the two images.
    ``'blend'`` computes the mean value.
    ``'checkerboard'`` makes tiles of dimension `n_tiles` that display
    alternatively the first and the second image.
    """
    if image2 is DEPRECATED:
        image2 = image1
        image1 = image0
    if image1.shape != image2.shape:
        raise ValueError('Images must have the same shape.')

    img1 = img_as_float(image1)
    img2 = img_as_float(image2)

    if method == 'diff':
        comparison = np.abs(img2 - img1)
    elif method == 'blend':
        comparison = 0.5 * (img2 + img1)
    elif method == 'checkerboard':
        shapex, shapey = img1.shape
        mask = np.full((shapex, shapey), False)
        stepx = int(shapex / n_tiles[0])
        stepy = int(shapey / n_tiles[1])
        for i, j in product(range(n_tiles[0]), range(n_tiles[1])):
            if (i + j) % 2 == 0:
                mask[i * stepx : (i + 1) * stepx, j * stepy : (j + 1) * stepy] = True
        comparison = np.zeros_like(img1)
        comparison[mask] = img1[mask]
        comparison[~mask] = img2[~mask]
    else:
        raise ValueError(
            'Wrong value for `method`. '
            'Must be either "diff", "blend" or "checkerboard".'
        )
    return comparison
