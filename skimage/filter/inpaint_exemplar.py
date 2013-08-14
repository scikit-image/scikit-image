from __future__ import division
import numpy as np
from skimage.util import img_as_float
from _inpaint_exemplar import _inpaint_criminisi


__all__ = ['inpaint_exemplar']


def inpaint_exemplar(source_image, synth_mask, window=9, max_thresh=0.2):
    """This function performs constrained synthesis. It grows the texture
    of surrounding region into the unknown pixels.

    Parameters
    ---------
    input_image : (M, N) array, uint8
        Input image whose texture is to be calculated
    synth_mask : (M, N) array, bool
        Texture for True values are to be synthesised
    window : int
        Size of the neighborhood window

    Returns
    -------
    image : (M, N) array, float
        Texture synthesised input_image

    References
    ---------
    .. [1] Criminisi, A., Pe ' ez, P., and Toyama, K. (2004). "Region filling
           and object removal by exemplar-based inpainting". IEEE Transactions
           on Image Processing, 13(9):1200-1212

    """

    source_image = img_as_float(source_image)

    h, w = source_image.shape
    offset = window // 2

    # Padding
    pad_size = (h + window - 1, w + window - 1)

    image = np.zeros(pad_size, dtype=np.float)
    mask = np.zeros(pad_size, np.uint8)

    image[offset:offset + h, offset:offset + w] = source_image
    mask[offset:offset + h, offset:offset + w] = synth_mask

    return _inpaint_criminisi(image, mask, window, max_thresh)
