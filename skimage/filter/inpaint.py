import numpy as np
from skimage.filter._inpaint_fmm import _fast_marching_method


__all__ = ['inpaint_fmm']


def inpaint_fmm(input_image, inpaint_mask, radius=5):
    """This function reconstructs masked regions of an image using the fast
    marching method to propagate information across the boundary between
    known and unknown regions.

    Image Inpainting technique based on the Fast Marching Method implementation
    as described in [1]_. FMM is used for computing the evolution of boundary
    moving in a direction *normal* to itself.

    Parameters
    ---------
    input_image : (M, N) array, unit8
        Grayscale image to be inpainted
    inpaint_mask : (M, N) array, bool
        Mask containing pixels to be inpainted. True values are inpainted
    radius : int
        Determining the range of the neighbourhood for inpainting a pixel

    Returns
    ------
    painted : (M, N) array, float
        The inpainted grayscale image

    Notes
    -----
    There are two main phases involved:
    - Initialisation - Refer to ``_init_fmm`` under the _inpaint_fmm.pyx file
    - Marching

    Marching Phase:
    Implementation under ``skimage.filter._inpaint._fast_marching_method``.
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
    >>> mask = np.zeros((8,8), np.uint8)
    >>> mask[2:-2, 2:-2] = 1
    >>> image = np.arange(64).reshape(8,8)
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

    """

    if input_image.shape[0] != inpaint_mask.shape[0]:
        raise TypeError("The first two dimensions of 'inpaint_mask' and "
                        "'input_image' do not match. ")
    if input_image.ndim > 1:
        if input_image.shape[1] != inpaint_mask.shape[1]:
            raise TypeError("The second dimension of 'inpaint_mask' and "
                            "'input_image' do not match. ")

    h, w = input_image.shape
    painted = np.zeros((h + 2, w + 2), np.float)
    mask = np.zeros((h + 2, w + 2), np.uint8)

    painted[1: -1, 1: -1] = input_image
    mask[1: -1, 1: -1] = inpaint_mask

    _fast_marching_method(painted, mask, radius=radius)

    return painted[1:-1, 1:-1]
