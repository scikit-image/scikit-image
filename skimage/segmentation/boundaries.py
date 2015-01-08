import numpy as np
from scipy import ndimage as nd
from ..morphology import dilation, erosion, square
from ..util import img_as_float
from ..color import gray2rgb
from .._shared.utils import deprecated


def find_boundaries(label_img, connectivity=1):
    """Return bool array where boundaries between labeled regions are True.

    Parameters
    ----------
    label_img : array of int
        An array in which different regions are labeled with different
        integers.
    connectivity: int in {1, ..., `label_img.ndim`}, optional
        A pixel is considered a boundary pixel if any of its neighbors
        has a different label. `connectivity` controls which pixels are
        considered neighbors. A connectivity of 1 (default) means
        pixels sharing an edge (in 2D) or a face (in 3D) will be
        considered neighbors. A connectivity of `label_img.ndim` means
        pixels sharing a corner will be considered neighbors.

    Returns
    -------
    boundaries : array of bool, same shape as `label_img`
        A bool image where `True` represents a boundary pixel.
    """
    selem = nd.generate_binary_structure(label_img.ndim, connectivity)
    boundaries = dilation(label_img, selem) != erosion(label_img, selem)
    return boundaries


def mark_boundaries(image, label_img, color=(1, 1, 0),
                    outline_color=(0, 0, 0)):
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
        outer_boundaries = dilation(boundaries.astype(np.uint8), square(3))
        image[outer_boundaries != 0, :] = np.array(outline_color)
    image[boundaries, :] = np.array(color)
    return image
