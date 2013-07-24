import numpy as np
import heapq
from skimage.morphology import dilation, disk

from skimage.filter._inpaint import fast_marching_method

__all__ = ['inpaint_fmm']


KNOWN = 0
BAND = 1
INSIDE = 2


def _init_fmm(mask):
    """Initialisation for Image Inpainting technique based on Fast Marching
    Method as outined in [1]_. Each pixel has 2 new values assigned to it
    stored in ``flag`` and ``u`` arrays.

    Parameters
    ----------
    _mask : 2D array of bool
        ``True`` values are to be inpainted.

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

    outside = dilation(mask, disk(1))
    band = np.logical_xor(mask, outside).astype(np.uint8)

    flag = (2 * outside) - band

    u = np.where(flag == INSIDE, 1.0e6, 0)

    heap = []
    indices = np.transpose(np.where(flag == BAND))
    for z in indices:
        heapq.heappush(heap, (u[tuple(z)], tuple(z)))

    return flag, u, heap


def inpaint_fmm(input_image, inpaint_mask, radius=5):
    """Inpaint image in areas specified by a mask. Image Inpainting technique
    based on the Fast Marching Method implementation as described in [1]_.
    FMM is used for computing the evolution of boundary moving in a direction
    *normal* to itself.

    Parameters
    ---------
    input_image : ndarray, np.uint8
        This can be either a single channel or three channel image.
    inpaint_mask : ndarray, bool
        Mask containing pixels to be inpainted. ``True`` values are inpainted.
    radius : int
        Determining the range of the neighbourhood for inpainting a pixel

    Returns
    ------
    painted : ndarray, np.uint8
        The inpainted image of same dimensions.

    Notes
    -----
    There are two main phases involved:
    - Initialisation
    - Marching

    Initialisation Phase:
    Implementaiton under ``skimage.filter.inpaint._init_fmm``.
    - ``flag`` Initialisation:
        All pixels are classified into 1 of the following flags:
        # 0 = KNOWN - intensity and u values are known.
        # 1 = BAND - u value undergoes an update.
        # 2 = INSIDE - intensity and u values unkown

    - ``u`` Initialisation:
        u <- 0 : ``flag`` equal to BAND or KNOWN
        u <- 1.0e6 (arbitrarily large value) : ``flag`` equal to INSIDE

    - ``heap`` Initialisation:
        Contains all the pixels marked as BAND in ``flag``. The heap element is
        a tuple with 2 elements, first being the ``u`` value corresponding to
        the tuple of index which is stored as the second element.
        Heap Element : (u[(i, j)], (i, j))

    Marching Phase:
    Implementation under ``skimage.filter._inpaint.fast_marching_method``.
    The steps of the algorithm are as follows:
    - Extract the pixel with the smallest ``u`` value in the BAND pixels
    - Update its ``flag`` value as KNOWN
    - March the boundary inwards by adding new points.
        - If they are either INSIDE or BAND, compute its ``u`` value using the
          ``eikonal`` function for all the 4 quadrants
        - If ``flag`` is INSIDE
            - Change it to BAND
            - Inpaint the pixel
        - Select the ``min`` value and assign it as ``u`` value of the pixel
        - Insert this new value in the ``heap``

    For further details, see [1]_

    References
    ---------
    .. [1] Telea, A., "An Image Inpainting Technique based on the Fast Marching
            Method", Journal of Graphic Tools (2004).
            http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from skimage.filter.inpaint import inpaint_fmm
    >>> image = np.arange(64).reshape(8,8)
    >>> image[2:-2, 2:-2] = 0
    >>> image
    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15],
           [16, 17,  0,  0,  0,  0, 22, 23],
           [24, 25,  0,  0,  0,  0, 30, 31],
           [32, 33,  0,  0,  0,  0, 38, 39],
           [40, 41,  0,  0,  0,  0, 46, 47],
           [48, 49, 50, 51, 52, 53, 54, 55],
           [56, 57, 58, 59, 60, 61, 62, 63]])
    >>> mask = np.zeros((8,8), np.uint8)
    >>> mask[2:-2, 2:-2] = 1
    >>> mask
    array([[0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> from skimage.filter.inpaint import inpaint_fmm
    >>> painted = inpaint_fmm(image, mask)
    >>> np.round(painted)
    array([[  0.,   1.,   2.,   3.,   4.,   5.,   6.,   7.],
           [  8.,   9.,  10.,  11.,  12.,  13.,  14.,  15.],
           [ 16.,  17.,  11.,  14.,  15.,  15.,  22.,  23.],
           [ 24.,  25.,  24.,  23.,  26.,  27.,  30.,  31.],
           [ 32.,  33.,  32.,  34.,  35.,  35.,  38.,  39.],
           [ 40.,  41.,  41.,  44.,  46.,  45.,  46.,  47.],
           [ 48.,  49.,  50.,  51.,  52.,  53.,  54.,  55.],
           [ 56.,  57.,  58.,  59.,  60.,  61.,  62.,  63.]])
    >>> plt.imshow(image); plt.show()
    >>> plt.imshow(painted); plt.show()

    """

    if inpaint_mask.ndim != 2:
        raise TypeError("The 'inpaint_mask' must be a two-dimensional array")

    h, w = input_image.shape
    painted = np.zeros((h + 2, w + 2), np.float)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    painted[1: -1, 1: -1] = input_image
    mask[1: -1, 1: -1] = inpaint_mask

    flag, u, heap = _init_fmm(mask)

    fast_marching_method(painted, flag, u, heap, radius=radius)

    return painted[1:-1, 1:-1]
