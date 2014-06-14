import numpy as np
from ..morphology import dilation, square
from ..util import img_as_float
from ..color import gray2rgb
from .._shared.utils import deprecated


def find_boundaries(label_img):
    """Return bool array where boundaries between labeled regions are True."""
    boundaries = np.zeros(label_img.shape, dtype=np.bool)
    boundaries[1:, :] += label_img[1:, :] != label_img[:-1, :]
    boundaries[:, 1:] += label_img[:, 1:] != label_img[:, :-1]
    return boundaries


def mark_boundaries(image, label_img, color=(1, 1, 0), outline_color=(0, 0, 0)):
    """Return image with boundaries between labeled regions highlighted.

    Parameters
    ----------
    image : (M, N[, 3]) array
        Grayscale or RGB image.
    label_img : (M, N) array
        Label array where regions are marked by different integer values.
    color : length-3 sequence
        RGB color of boundaries in the output image.
    outline_color : length-3 sequence
        RGB color surrounding boundaries in the output image. If None, no
        outline is drawn.
    """
    if image.ndim == 2:
        image = gray2rgb(image)
    image = img_as_float(image, force_copy=True)

    boundaries = find_boundaries(label_img)
    if outline_color is not None:
        outer_boundaries = dilation(boundaries.astype(np.uint8), square(2))
        image[outer_boundaries != 0, :] = np.array(outline_color)
    image[boundaries, :] = np.array(color)
    return image
