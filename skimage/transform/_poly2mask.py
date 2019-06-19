import numpy as np
import matplotlib

from .. import draw


def poly2mask(image_shape, polygon, backend='default'):
    """Compute a mask from polygon.

    Parameters
    ----------
    image_shape : tuple of size 2.
        The shape of the mask.
    polygon : array_like.
        The polygon coordinates of shape (N, 2) where N is
        the number of points.
    backend : 'default' or 'matplotlib'.
        The default implementation uses `skimage.draw`, while the
        'matplotlib' implementtion uses `maplotlib.path.Path`.
        While note exactly the same both methods give near-identical results.

    Returns
    -------
    mask : 2-D ndarray of type 'bool'.
        The mask that corresponds to the input polygon.
    """
    if backend == 'default':
        return _poly2mask_skimage(image_shape, polygon)
    elif backend == 'matplotlib':
        return _poly2mask_mpl(image_shape, polygon)

    raise ValueError(
        "Wrong backend selected. Choose either 'default' or 'matplotlib'.")


def _poly2mask_skimage(image_shape, polygon):
    polygon = np.array(polygon)
    fill_row_coords, fill_col_coords = draw.polygon(
        polygon[:, 1], polygon[:, 0], image_shape)
    mask = np.zeros(image_shape, dtype=np.bool)
    mask[fill_row_coords, fill_col_coords] = True
    return mask


def _poly2mask_mpl(image_shape, polygon):
    polygon = np.array(polygon)
    xx, yy = np.meshgrid(np.arange(image_shape[1]), np.arange(image_shape[0]))
    xx, yy = xx.flatten(), yy.flatten()
    indices = np.vstack((xx, yy)).T
    mask = matplotlib.path.Path(polygon).contains_points(indices)
    mask = mask.reshape(image_shape)
    mask = mask.astype(np.bool)
    return mask
