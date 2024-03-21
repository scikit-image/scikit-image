import functools
import inspect
import warnings
from itertools import product

import numpy as np

from .dtype import img_as_float
from skimage._shared.utils import (
    DEPRECATED,
)


class _rename_image_params:
    """Deprecate parameters `image1, image2` in favour of `image0, image1` in
    function `compare_images`.

    Parameters
    ----------
    deprecated_name : str
        The name of the deprecated parameter.
    start_version : str
        The package version in which the warning was introduced.
    stop_version : str
        The package version in which the warning will be replaced by
        an error / the deprecation is completed.
    """

    def __init__(
        self,
        deprecated_name,
        *,
        start_version,
        stop_version,
    ):
        self.deprecated_name = deprecated_name
        self.start_version = start_version
        self.stop_version = stop_version

    def __call__(self, func):
        parameters = inspect.signature(func).parameters
        if parameters['image2'].default is not DEPRECATED:
            raise RuntimeError(
                f"Expected `{self.deprecated_name}` to have the value {DEPRECATED!r} "
                f"to indicate its status in the rendered signature."
            )
        warning_message = (
            "Since version 0.23, the two input images are named `image0` and "
            "`image1` (instead of `image1` and `image2`, respectively). Please use "
            "`image0, image1` to avoid this warning for now, and avoid an error "
            "from version 0.25 onwards."
        )
        wm_method = (
            "Starting in version 0.25, all arguments following `image0, image1` "
            "(including `method`) will be keyword-only. Please pass `method=` "
            "in the function call to avoid this warning for now, and avoid an error "
            "from version 0.25 onwards."
        )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if 'image2' not in kwargs.keys():
                kwargs['image2'] = DEPRECATED

            # Pass first all args as kwargs
            if len(args) > 0:
                kwargs['image0'] = args[0]
                if len(args) > 1:
                    kwargs['image1'] = args[1]
                    if len(args) > 2 and args[len(args) - 1] in [
                        'diff',
                        'blend',
                        'checkerboard',
                    ]:
                        warnings.warn(wm_method, category=FutureWarning)
                        kwargs['method'] = args[len(args) - 1]

            if kwargs['image2'] is not DEPRECATED:
                deprecated_value = kwargs['image2']
                kwargs['image2'] = DEPRECATED
                if 'image1' in kwargs.keys():
                    if 'image0' in kwargs.keys():
                        raise RuntimeError(
                            "Use `image0, image1` to pass the two input images."
                        )
                    else:
                        warnings.warn(warning_message, category=FutureWarning)
                        args = (kwargs['image1'], deprecated_value)
                else:
                    if 'image0' in kwargs.keys():
                        warnings.warn(warning_message, category=FutureWarning)
                        args = (kwargs['image0'], deprecated_value)

            kwargs.pop('image2')
            if 'image0' in kwargs.keys():
                kwargs.pop('image0')
            if 'image1' in kwargs.keys():
                kwargs.pop('image1')
            return func(*args, **kwargs)

        return wrapper


@_rename_image_params("image2", start_version="0.23", stop_version="0.25")
def compare_images(image0, image1, image2=DEPRECATED, method='diff', *, n_tiles=(8, 8)):
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

    if image1.shape != image0.shape:
        raise ValueError('Images must have the same shape.')

    img1 = img_as_float(image0)
    img2 = img_as_float(image1)

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
