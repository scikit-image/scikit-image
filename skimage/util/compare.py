import warnings
from itertools import product

import numpy as np

from .dtype import img_as_float
from skimage._shared.utils import (
    DEPRECATED,
)


def compare_images(
    image0=None, image1=None, image2=DEPRECATED, *, method='diff', n_tiles=(8, 8)
):
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
    warning_message = (
        "Since version 0.23, the two input images are named "
        "`image0` and `image1` (instead of `image1` and `image2`). Please use "
        "`image0, image1` to avoid this warning for now, and avoid an error "
        "from version 0.25 onwards."
    )
    if image0 is None:
        if image1 is None:
            raise ValueError("You must pass two input images.")
        else:
            warnings.warn(warning_message, category=FutureWarning)
    elif image2 is DEPRECATED:
        image2 = image1
        image1 = image0
        warnings.warn(warning_message, category=FutureWarning)
    else:
        if image1 is None:
            image1 = image0
            warnings.warn(warning_message, category=FutureWarning)
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
