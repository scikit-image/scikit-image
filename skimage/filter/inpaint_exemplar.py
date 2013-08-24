from __future__ import division
import numpy as np
from skimage.util import img_as_float
from skimage.filter._inpaint_exemplar import _inpaint_criminisi


__all__ = ['inpaint_exemplar']


def inpaint_exemplar(source_image, synth_mask, window, max_thresh=0.2):
    """This function performs constrained synthesis using Criminisi et al.
    algorithm. It grows the texture of surrounding region into the unknown
    pixels. See Notes for an outline of the algorithm.

    Parameters
    ---------
    source_image : (M, N) array, uint8
        Input image whose texture is to be calculated
    synth_mask : (M, N) array, bool
        Texture for True values are to be synthesised
    window : int
        Size of the neighborhood window, refer to Notes below for details on
        choice of value
    max_thresh : float, optional
        Amount of threshold allowed for template matching

    Returns
    -------
    image : (M, N) array, float
        Texture synthesised input_image

    Notes
    -----
    For best results, `window` should be larger in size than the texel (texture
    element) being inpainted. For example, in the case below of
    `skimage.data.checkerboard` image, the single white/black square is the
    texel which is of `(25, 25)` shape. A value larger than this yields
    perfect reconstruction, but a value smaller than this, may have couple of
    pixels off.

    Outline of the algorithm for Texture Synthesis is as follows:
    - Loop: Generate the boundary pixels of the region to be inpainted
        - Loop: Compute the priority of each pixel
            - Generate a template of (window, window), center: boundary pixel
            - confidence_term: avg amount of reliable information in template
            - data_term: strength of the isophote hitting this boundary pixel
            - priority = data_term * confidence_term
        - Repeat for all boundary pixels and chose the pixel with max priority
        - Template matching of the pixel with max priority
            - Generate a template of (window, window) around this pixel
            - Compute the SSD between template and similar sized patches across
              the image
            - Find the pixel with smallest SSD, such that patch isn't where
              template is located (False positive)
            - Update the intensity value of the unknown region of template as
              the corresponding value from matched patch
    - Repeat until all pixels are inpainted

    For further information refer to [1]_

    References
    ---------
    .. [1] Criminisi, A., Pe ' ez, P., and Toyama, K. (2004). "Region filling
           and object removal by exemplar-based inpainting". IEEE Transactions
           on Image Processing, 13(9):1200-1212

    Example
    -------
    >>> import numpy as np
    >>> from skimage.data import checkerboard
    >>> from skimage.filter.inpaint_exemplar import inpaint_exemplar
    >>> image = checkerboard().astype(np.uint8)
    >>> mask = np.zeros_like(image, dtype=np.uint8)
    >>> paint_region = (slice(75, 125), slice(75, 125))
    >>> image[paint_region] = 0
    >>> mask[paint_region] = 1
    >>> painted = inpaint_exemplar(image, mask, window=27, max_thresh=0.2)

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
