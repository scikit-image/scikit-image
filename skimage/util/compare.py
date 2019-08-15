import numpy as np
from ..util import img_as_float64
from itertools import product


def compare_images(image1, image2, method='diff', *, n_tiles=(8, 8)):
    """
    Return an image showing the differences between two images.

    .. versionadded:: 0.16

    Parameters
    ----------
    image1, image2 : 2-D array
        Images to process, must be of the same shape.
    method : string, optional
        Method used for the comparison.
        Valid values are {'diff', 'blend', 'checkerboard'}
    n_tiles : tuple, optional
        Used only for the `checkerboard` method. Specifies the number
        of tiles (row, column) to divide the image.

    Returns
    -------
    comparison : 2-D array
        Image showing the differences.
    """
    img1 = img_as_float64(image1)
    img2 = img_as_float64(image2)

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
            if (i+j) % 2 == 0:
                mask[i*stepx:(i+1)*stepx, j*stepy:(j+1)*stepy] = True
        comparison = np.zeros_like(img1)
        comparison[mask] = img1[mask]
        comparison[~mask] = img2[~mask]
    else:
        raise ValueError('Wrong value for `method`. '
                         'Must be either "diff", "blend" or "checkerboard".')
    return comparison
