import numpy as np
from ..util import img_as_float
from itertools import product


def compare_images(
    image1, image2, method="diff", *, n_tiles=(8, 8), channel_axis: int = -1
):
    """
    Return an image showing the differences between two images.

    .. versionadded:: 0.16

    Parameters
    ----------
    image1, image2 : 2-D array
        Images to process, must be of the same shape.
    method : string, optional
        Method used for the comparison.
        Valid values are {'diff', 'blend', 'checkerboard'}.
        Details are provided in the note section.
    n_tiles : tuple, optional
        Used only for the `checkerboard` method. Specifies the number
        of tiles (row, column) to divide the image.
    channel_axis : int, optional
        Defines the channel axis of the input images. By default, the last
        axis is considered as the channel axis.

    Returns
    -------
    comparison : 2-D array
        Image showing the differences.

    Notes
    -----
    ``'diff'`` computes the absolute difference between the two images.
    ``'blend'`` computes the mean value.
    ``'checkerboard'`` makes tiles of dimension `n_tiles` that display
    alternatively the first and the second image.
    """
    if image1.shape != image2.shape:
        raise ValueError("Images must have the same shape.")
    if image1.ndim < 2 or image1.ndim > 3:
        raise ValueError(
            "Images must be 2-D with a single channel or 3-D with a channel dimension. Ndarrays are currently not supported."
        )
    if not isinstance(channel_axis, int):
        raise TypeError("channel_axis must be an integer")
    if channel_axis < -image1.ndim or channel_axis >= image1.ndim:
        raise np.AxisError("channel_axis exceeds image dimensions")

    img1 = img_as_float(np.moveaxis(image1, source=channel_axis, destination=-1))
    img2 = img_as_float(np.moveaxis(image2, source=channel_axis, destination=-1))

    if method == "diff":
        comparison = np.abs(img2 - img1)
    elif method == "blend":
        comparison = 0.5 * (img2 + img1)
    elif method == "checkerboard":
        shapex, shapey = img1.shape[:2]

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
            "Wrong value for `method`. "
            'Must be either "diff", "blend" or "checkerboard".'
        )

    # restore change to channel axis
    comparison = np.moveaxis(comparison, source=-1, destination=channel_axis)

    return comparison
