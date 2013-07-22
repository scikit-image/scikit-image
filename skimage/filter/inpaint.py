import numpy as np
import heapq
from skimage.morphology import dilation, disk

from _inpaint import fast_marching_method

__all__ = ['inpaint_fmm']


KNOWN = 0
BAND = 1
INSIDE = 2


def _init_fmm(_mask):
    """Initialisation for Image Inpainting technique based on Fast Marching
    Method as outined in [1]_. Each pixel has 2 new values assigned to it
    stored in `flag` and `u` arrays.

    Parameters
    ----------
    _mask : 2D array of bool
        `True` values are to be inpainted.

    Returns
    ------
    flag : array of int
        Array marking pixels as known, along the boundary to be solved, or
        inside the unknown region: 0 = KNOWN, 1 = BAND, 2 = INSIDE
    u : array of float
        The distance/time map from the boundary to each pixel.
    heap : list of tuples
        BAND points with heap element as mentioned above

    Notes
    -----
    ``flag`` Initialisation:
    All pixels are classified into 1 of the following flags:
    # 0 = KNOWN - intensity and u values are known.
    # 1 = BAND - u value undergoes an update.
    # 2 = INSIDE - intensity and u values unkown

    ``u`` Initialisation:
    u <- 0 : ``flag`` equal to BAND or KNOWN
    u <- 1.0e6 (arbitrarily large value) : ``flag`` equal to INSIDE

    ``heap`` Initialisation:
    Contains all the pixels marked as BAND in ``flag``. The heap element is
    a tuple with 2 elements, first being the ``u`` value corresponding to the
    tuple of index which is stored as the second element.
    Heap Element : (u[(i, j)], (i, j))

    References
    ----------
    .. [1] Telea, A., "An Image Inpainting Technique based on the Fast Marching
            Method", Journal of Graphic Tools (2004).
            http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    """

    mask = _mask.astype(np.uint8)
    outside = dilation(mask, disk(1))
    band = np.logical_xor(mask, outside).astype(np.uint8)

    flag = (2 * outside) - band

    u = np.where(flag == INSIDE, 1.0e6, 0)

    heap = []
    indices = np.transpose(np.where(flag == BAND))
    for z in indices:
        heapq.heappush(heap, (u[tuple(z)], tuple(z)))

    return flag, u, heap


def inpaint_fmm(input_image, inpaint_mask, _run_inpaint=True, neighbor=5):
    """Inpaint image in areas specified by a mask.

    Parameters
    ---------
    input_image : array
        This can be either a single channel or three channel image.
    inpaint_mask : array, bool
        Mask containing pixels to be inpainted. `True` values are inpainted.
    neighbor : int
        Determining the range of the neighbourhood for inpainting a pixel

    Returns
    ------
    painted : array
        The inpainted image.

    References
    ---------
    .. [1] Telea, A., "An Image Inpainting Technique based on the Fast Marching
            Method", Journal of Graphic Tools (2004).
            http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    """
    # TODO: Error checks. Image either 3 or 1 channel. All dims same

    h, w = input_image.shape
    image = np.zeros((h + 2, w + 2), np.uint8)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    image[1: -1, 1: -1] = input_image
    mask[1: -1, 1: -1] = inpaint_mask

    flag, u, heap = _init_fmm(mask)

    fast_marching_method(image, flag, u, heap, _run_inpaint, neighbor=neighbor)

    return image[1:-1, 1:-1]
