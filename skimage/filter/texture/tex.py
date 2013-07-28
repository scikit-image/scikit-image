import numpy as np
from skimage.morphology import erosion, disk
from numpy.lib.stride_tricks import as_strided
from skimage.util import img_as_float


def inpaint_texture(input_image, synth_mask, window=3, max_thresh=0.2):
    """This function performs constrained synthesis. It grows the texture
    of surrounding region into the unknown pixels.

    Parameters
    ---------
    input_image : (M, N) array, np.uint8
        Input image whose texture is to be calculated
    synth_mask : (M, N) array, bool
        Texture for `True` values are to be synthesised.
    window : int
        Size of the neighborhood window

    Returns
    -------
    image : array, float
        Texture synthesised input_image.

    References
    ---------
    .. [1] A. Efros and T. Leung. "Texture Synthesis by Non-Parametric
            Sampling". In Proc. Int. Conf. Computer Vision, pages 1033-1038,
            Kerkyra, Greece, September 1999.
            http://graphics.cs.cmu.edu/people/efros/research/EfrosLeung.html

    """

    input_image[synth_mask] = 0

    h, w = input_image.shape
    offset = window / 2

    # Padding
    pad_size = (h + window - 1, w + window - 1)
    image = (np.mean(input_image) / 255.) * np.ones(pad_size, dtype=np.uint8)
    mask = np.zeros(pad_size, np.uint8)

    image[offset:offset + h, offset:offset + w] = img_as_float(input_image)
    mask[offset:offset + h, offset:offset + w] = synth_mask
    ssd = np.zeros(input_image.shape, np.float)

    t_row, t_col = np.ogrid[(-offset):(offset + 1), (-offset):(offset + 1)]
    sigma = window / 6.4
    gauss_mask = _gaussian(sigma, (window, window))

    while mask.any():
        progress = 0

        # Generate the boundary of ROI (region to be synthesised)
        boundary = mask - erosion(mask, disk(1))
        if not boundary.any():  # If the remaining region is 1-pixel thick
            boundary = mask

        bound_list = np.transpose(np.where(boundary == 1))

        for i_b, j_b in bound_list:
            template = image[i_b + t_row, j_b + t_col]
            mask_template = mask[i_b + t_row, j_b + t_col]
            valid_mask = gauss_mask * (1 - mask_template)

            ssd = sum_sq_diff(image, template, valid_mask)

            # Remove the case where sample == template
            ssd[i_b - offset, j_b - offset] = 1.

            matched_index = np.transpose(np.where(ssd == ssd.min()))[0]

            if ssd[tuple(matched_index)] < max_thresh:
                image[i_b, j_b] = image[tuple(matched_index + [window / 2,
                                                               window / 2])]
                mask[i_b, j_b] = False
                progress = 1

        if progress == 0:
            max_thresh = 1.1 * max_thresh

    return image[offset:-offset, offset:-offset]


def sum_sq_diff(input_image, template, valid_mask):
    total_weight = valid_mask.sum()
    window_size = template.shape
    y = as_strided(input_image,
                   shape=(input_image.shape[0] - window_size[0] + 1,
                          input_image.shape[1] - window_size[1] + 1,) +
                   window_size,
                   strides=input_image.strides * 2)
    ssd = np.einsum('ijkl, kl, kl->ij', y, template, valid_mask, dtype=np.float)
    ssd *= - 2
    ssd += np.einsum('ijkl, ijkl, kl->ij', y, y, valid_mask)
    ssd += np.einsum('ij, ij, ij', template, template, valid_mask)
    return ssd / total_weight


def _gaussian(sigma=0.5, size=None):
    """Gaussian kernel numpy array with given sigma and shape.
    """
    sigma = max(abs(sigma), 1e-10)

    x = np.arange(-(size[0] - 1) / 2.0, (size[0] - 1) / 2.0 + 1e-8)
    y = np.arange(-(size[1] - 1) / 2.0, (size[1] - 1) / 2.0 + 1e-8)

    Kx = np.exp(-x ** 2 / (2 * sigma ** 2))
    Ky = np.exp(-y ** 2 / (2 * sigma ** 2))
    ans = np.outer(Kx, Ky) / (2.0 * np.pi * sigma ** 2)
    return ans / ans.sum()
