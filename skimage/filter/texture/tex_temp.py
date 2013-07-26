import numpy as np
from skimage.morphology import erosion, disk
from skimage.feature import match_template
from skimage.util import img_as_float


def grow_image(image, syn_mask, window):
    """This function performs constrained synthesis. It grows the texture
    of surrounding region into the unknown pixels.

    Parameters
    ---------
    image : (M, N) array, np.uint8
        Input image whose texture is to be calculated
    syn_mask : (M, N) array, bool
        Texture for `True` values are to be synthesised.
    window : int
        Size of the neighborhood window

    Returns
    -------
    image : array, float
        Texture synthesised image.

    References
    ---------
    .. [1] Criminisi, A., Pe ́rez, P., and Toyama, K. (2004). "Region filling
            and object removal by exemplar-based inpainting". IEEE Transactions
            on Image Processing, 13(9):1200–1212.
            http://graphics.cs.cmu.edu/people/efros/research/EfrosLeung.html

    """

    max_thresh = 0.2

    image = img_as_float(image)

    sigma = window / 6.4
    gauss_mask = _gaussian(sigma, (window, window))

    while syn_mask.any():
        progress = 0

        # Generate the boundary of ROI (region to be synthesised)
        boundary = syn_mask - erosion(syn_mask, disk(1))
        if not boundary.any():  # If the remaining region is 1-pixel thick
            boundary = syn_mask

        bound_list = np.transpose(np.where(boundary == 1))

        for i_b, j_b in bound_list:
            template = image[(i_b - window / 2):(i_b + window / 2 + 1),
                             (j_b - window / 2):(j_b + window / 2 + 1)]
            mask_template = syn_mask[(i_b - window / 2):(i_b + window / 2 + 1),
                                     (j_b - window / 2):(j_b + window / 2 + 1)]
            valid_template = (1 - mask_template) * template * gauss_mask
            valid_template /= np.sum(gauss_mask)

            corr = match_template(image, valid_template, pad_input=True)
            # Remove the case where sample == template
            corr[i_b, j_b] = 0.

            best_matches = np.transpose(np.where(corr == corr.max()))[0]

            matched_index = tuple(best_matches)

            if corr[matched_index] < max_thresh:
                image[i_b, j_b] = image[matched_index]
                syn_mask[i_b, j_b] = False
                progress = 1

        if progress == 0:
            max_thresh = 1.1 * max_thresh

    return image


def _gaussian(sigma=0.5, size=None):
    """Gaussian kernel numpy array with given sigma and shape.
    """
    sigma = max(abs(sigma), 1e-10)

    x = np.arange(-(size[0] - 1) / 2.0, (size[0] - 1) / 2.0 + 1e-8)
    y = np.arange(-(size[1] - 1) / 2.0, (size[1] - 1) / 2.0 + 1e-8)

    Kx = np.exp(-x ** 2 / (2 * sigma ** 2))
    Ky = np.exp(-y ** 2 / (2 * sigma ** 2))
    ans = np.outer(Kx, Ky) / (2.0 * np.pi * sigma ** 2)
    return ans / sum(sum(ans))
