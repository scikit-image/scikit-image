import numpy as np
from ._inpaint_fmm import _inpaint_fmm
from ..util import img_as_float


__all__ = ['inpaint_fmm']


def inpaint_fmm(image, mask, radius=5):
    """Inpaint image using Fast Marching Method.

    Image inpainting technique based on the Fast Marching Method (FMM)
    implementation as described in [1]_. FMM is used for computing the
    evolution of boundary moving in a direction perpendicular to itself.

    Parameters
    ---------
    image : (M, N) array
        Grayscale image to be inpainted.
    mask : (M, N) array, bool
        Non-zero values denoting regions to be inpainted.
    radius : int
        Determining the range of the neighborhood for inpainting a pixel.

    Returns
    ------
    inpainted : (M, N) array, float
        The inpainted grayscale image.

    References
    ---------
    .. [1] Telea, A., "An Image Inpainting Technique based on the Fast Marching
           Method", Journal of Graphic Tools (2004).
           http://iwi.eldoc.ub.rug.nl/FILES/root/2004/JGraphToolsTelea/2004JGraphToolsTelea.pdf

    Examples
    --------
    >>> from skimage.restoration import inpaint_fmm
    >>> mask = np.zeros((8, 8), dtype=np.bool)
    >>> mask[2:-2, 2:-2] = True
    >>> image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    >>> image[mask == 1] = 0
    >>> image
    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
           [ 8,  9, 10, 11, 12, 13, 14, 15],
           [16, 17,  0,  0,  0,  0, 22, 23],
           [24, 25,  0,  0,  0,  0, 30, 31],
           [32, 33,  0,  0,  0,  0, 38, 39],
           [40, 41,  0,  0,  0,  0, 46, 47],
           [48, 49, 50, 51, 52, 53, 54, 55],
           [56, 57, 58, 59, 60, 61, 62, 63]])
    >>> inpaint_fmm(image, mask)
    array([[ 0,  1,  2,  3,  4,  5,  6,  7],
       [ 8,  9, 10, 11, 12, 13, 14, 15],
       [16, 17, 11, 13, 15, 15, 22, 23],
       [24, 25, 23, 22, 26, 26, 30, 31],
       [32, 33, 31, 33, 33, 34, 38, 39],
       [40, 41, 40, 44, 45, 44, 46, 47],
       [48, 49, 50, 51, 52, 53, 54, 55],
       [56, 57, 58, 59, 60, 61, 62, 63]], dtype=uint8)

    """

    if image.shape != mask.shape:
        raise ValueError("The dimensions of `mask` and `image` do not match. ")

    if not np.issubdtype(image.dtype, np.uint8):
        raise TypeError("`image` must be of dtype `uint8`.")

    if radius < 1:
        raise ValueError("`radius` must be greater than 0.")

    rows, cols = image.shape
    inpainted = np.zeros((rows + 2, cols + 2), dtype=np.double)
    inpainted_mask = np.zeros((rows + 2, cols + 2), dtype=np.uint8)
    inner = (slice(1, -1), slice(1, -1))
    inpainted[inner] = image
    inpainted_mask[inner] = mask

    _inpaint_fmm(inpainted, inpainted_mask, radius=radius)

    # Make sure all values are in valid range for uint8
    np.clip(inpainted, 0, 255, inpainted)

    # Convert back to uint8 and original size
    inpainted = np.round(inpainted[inner]).astype(np.uint8)

    return inpainted
