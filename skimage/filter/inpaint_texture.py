from __future__ import division
import numpy as np
from skimage.util import img_as_float
from skimage.filter._inpaint_texture import _inpaint_efros


__all__ = ['inpaint_texture']


def inpaint_texture(source_image, synth_mask, window=5, max_thresh=0.2):
    """This function performs constrained texture synthesis. It grows the
    texture of surrounding region into the unknown pixels. This implementation
    is pixel-based. Check the Notes Section for a brief overview of the
    algorithm.

    Parameters
    ---------
    source_image : (M, N) array, uint8
        Input image whose texture is to be calculated
    synth_mask : (M, N) array, bool
        Texture for True values are to be synthesised
    window : int
        Size of the neighborhood window, (window, window)
    max_thresh : float
        Maximum tolerable SSD (Sum of Squared Difference) between the template
        around a pixel to be filled and an equal size image sample

    Returns
    -------
    painted : array, np.float
        Texture synthesised image

    Notes
    -----
    Outline of the algorithm for Texture Synthesis is as follows:
    - Loop: Generate the boundary pixels of the region to be inpainted
        - Loop: Generate a template of (window, window), center: boundary pixel
            - Compute the SSD between template and similar sized patches across
              the image
            - Find the pixel with smallest SSD, such that patch isn't where
              template is located (False positive)
            - Update the intensity value of center pixel of template as the
              value of the center of the matched patch
        - Repeat for all pixels of the boundary
    - Repeat until all pixels are inpainted

    For further information refer to [1]_

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
