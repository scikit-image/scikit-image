import numpy as np
from scipy import ndimage as nd
from ..morphology import dilation, erosion, square
from ..util import img_as_float
from ..color import gray2rgb
from .._shared.utils import deprecated


def find_boundaries(label_img, connectivity=1, mode='thick', background=0):
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
    mode: string in {'thick', 'inner', 'outer', 'subpixel'}
        How to mark the boundaries:

        - thick: any pixel not completely surrounded by pixels of the
        same label (defined by `connectivity`) is marked as a boundary.
        This results in boundaries that are 2 pixels thick.
        - inner: outline the pixels *just inside* of objects, leaving
        background pixels untouched.
        - outer: outline pixels in the background around object
        boundaries. When two objects touch, their boundary is also
        marked.
        - subpixel: return a doubled image, with pixels *between* the
        original pixels marked as boundary where appropriate.
    background: int, optional
        For modes 'inner' and 'outer', a definition of a background
        label is required. See `mode` for descriptions of these two.

    Returns
    -------
    boundaries : array of bool, same shape as `label_img`
        A bool image where ``True`` represents a boundary pixel. For
        `mode` equal to 'subpixel', ``boundaries.shape[i]`` is equal
        to ``2 * label_img.shape[i] - 1`` for all ``i`` (a pixel is
        inserted in between all other pairs of pixels).

    Examples
    --------
    >>> labels = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
    ...                    [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
    ...                    [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
    ...                    [0, 0, 1, 1, 1, 5, 5, 5, 0, 0],
    ...                    [0, 0, 0, 0, 0, 5, 5, 5, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ...                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)
    >>> find_boundaries(labels).astype(np.uint8) # display 1/0, not True/False
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
           [0, 1, 1, 0, 1, 1, 0, 1, 1, 0],
           [0, 1, 1, 1, 1, 1, 0, 1, 1, 0],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> find_boundaries(labels, mode='inner').astype(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
           [0, 0, 1, 0, 1, 1, 0, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> find_boundaries(labels, mode='outer').astype(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
           [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
           [0, 0, 1, 1, 1, 1, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> find_boundaries(labels, mode='subpixel').astype(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    ndim = label_img.ndim
    selem = nd.generate_binary_structure(ndim, connectivity)
    if mode != 'subpixel':
        boundaries = dilation(label_img, selem) != erosion(label_img, selem)
        if mode == 'inner':
            foreground_image = (label_img != background)
            boundaries &= foreground_image
        elif mode == 'outer':
            background_image = (label_img == background)
            selem = nd.generate_binary_structure(ndim, ndim)
            no_adjacent_background = ~dilation(background_image, selem)
            boundaries &= (background_image | no_adjacent_background)
        return boundaries
    else:
        label_img_expanded = np.zeros([(2 * s - 1) for s in label_img.shape],
                                      label_img.dtype)
        pixels = [slice(None, None, 2)] * ndim
        selem = nd.generate_binary_structure(ndim, ndim)
        label_img_expanded[pixels] = label_img
        max_label = np.iinfo(label_img.dtype).max
        label_img_edge_inverted = np.array(label_img_expanded, copy=True)
        label_img_edge_inverted[label_img_expanded == 0] = max_label
        boundaries = (dilation(label_img_expanded, selem) !=
                      erosion(label_img_edge_inverted, selem))
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
