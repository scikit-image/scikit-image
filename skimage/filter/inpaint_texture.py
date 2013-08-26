from __future__ import division
import numpy as np
from skimage.util import img_as_float
from skimage.filter._inpaint_efros import _inpaint_efros


__all__ = ['inpaint_efros']


def inpaint_efros(source_image, synth_mask, window=5, max_thresh=0.2):
    """Returns the image with the masked region painted in.

    This function performs constrained texture synthesis. It grows the
    texture of surrounding region into the unknown pixels. This implementation
    updates pixel-by-pixel.

    Parameters
    ---------
    source_image : (M, N) array, uint8
        Input image whose texture is to be calculated.
    synth_mask : (M, N) array, bool
        Texture for True values are to be synthesised.
    window : int, optional
        Width of the neighborhood window. (window, window) patch about the
        pixel to be inpainted. Preferably odd, for symmetry.
    max_thresh : float, optional
        Maximum tolerable SSD (Sum of Squared Difference) between the template
        around a pixel to be filled and an equal size image sample.

    Returns
    -------
    painted : array, float
        Texture synthesised image.

    References
    ---------
    .. [1] A. Efros and T. Leung. "Texture Synthesis by Non-Parametric
           Sampling". In Proc. Int. Conf. Computer Vision, pages 1033-1038,
           Kerkyra, Greece, September 1999.
           http://graphics.cs.cmu.edu/people/efros/research/EfrosLeung.html

    Example
    -------
    >>> import numpy as np
    >>> from skimage.filter.inpaint_texture import inpaint_texture
    >>> from skimage.data import checkerboard
    >>> image = np.round(checkerboard()[92:108, 92:108])
    >>> mask = np.zeros_like(image, np.uint8)
    >>> mask[5:-5, 5:-5] = 1
    >>> image[mask == 1] = 0
    >>> painted = inpaint_texture(image, mask, window=5)

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

    return _inpaint_efros(image, mask, window, max_thresh)
