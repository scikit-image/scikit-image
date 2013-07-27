import numpy as np
from skimage.morphology import erosion, disk


def inpaint_exemplar(input_image, synth_mask, window=9):
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
    .. [1] Criminisi, A., Pe ' ez, P., and Toyama, K. (2004). "Region filling
            and object removal by exemplar-based inpainting".IEEE Transactions
            on Image Processing, 13(9):1200-1212

    """

    max_thresh = 0.2
    input_image[synth_mask] = 0

    h, w = input_image.shape
    offset = window / 2

    # Initialization and Padding
    pad_size = (h + window - 1, w + window - 1)
    image = input_image.mean() * np.ones(pad_size, dtype=np.uint8)
    mask = np.zeros(pad_size, np.uint8)
    image_grad_y = np.zeros((h + window - 3, w + window - 3), np.int16)
    image_grad_x = np.zeros((h + window - 3, w + window - 3), np.int16)
    nx = np.zeros((h + window - 3, w + window - 3), np.int8)
    ny = np.zeros((h + window - 3, w + window - 3), np.int8)
    ssd = np.zeros((h, w), np.float)

    image[offset:offset + h, offset:offset + w] = input_image
    mask[offset:offset + h, offset:offset + w] = synth_mask
    confidence = 1. - mask

    t_row, t_col = np.ogrid[(-offset):(offset + 1), (-offset):(offset + 1)]
    sigma = window / 6.4
    gauss_mask = _gaussian(sigma, (window, window))

    while mask.any():
        # Generate the fill_front, boundary of ROI (region to be synthesised)
        fill_front = mask - erosion(mask, disk(1))
        if not fill_front.any():  # If the remaining region is 1-pixel thick
            fill_front = mask

        # Generate the image gradient and normal vector to the boundary
        image_grad_y = image[2:, 1:-1] - image[:-2, 1:-1]
        image_grad_x = image[1:-1, 2:] - image[1:-1, :-2]
        ny = fill_front[2:, 1:-1] - fill_front[:-2, 1:-1]
        nx = fill_front[1:-1, 2:] - fill_front[1:-1, :-2]

        # Generate the indices of the pixels in fill_front
        fill_front_indices = np.transpose(np.where(fill_front == 1))

        max_priority, max_conf, i_max, j_max = 0, 0, 0, 0

        # Determine the priority of pixels on the boundary, hence the order
        for k in xrange(fill_front_indices.shape[0]):
            i = fill_front_indices[k, 0]
            j = fill_front_indices[k, 1]

            # Compute the confidence term
            confidence_term = confidence[i + t_row, j + t_col].sum() / (
                window ** 2)

            # Compute the data term
            temp_grad_x = image_grad_x[(i - 1) + t_row, (j - 1) + t_col]
            temp_grad_y = image_grad_y[(i - 1) + t_row, (j - 1) + t_col]
            mod_grad = (temp_grad_x ** 2 + temp_grad_y ** 2)
            ind_max = tuple(np.transpose(np.where(
                mod_grad == mod_grad.max()))[0])
            data_term = abs(temp_grad_x[ind_max] * nx[ind_max] - temp_grad_y[
                ind_max] * ny[ind_max])
            data_term /= mod_grad[ind_max] * (nx[ind_max] ** 2 + ny[
                ind_max] ** 2)

            # Compute the priority for determining the order for inpainting
            priority = data_term * confidence_term
            if priority > max_priority:
                max_priority = priority
                max_conf = confidence_term
                i_max, j_max = i, j

        template = image[i_max + t_row, j_max + t_col]
        mask_template = mask[i_max + t_row, j_max + t_col]
        valid_mask = gauss_mask * (1 - mask_template)

        total_weight = valid_mask.sum()
        for i in xrange(h):
            for j in xrange(w):
                patch = image[i + t_row, j + t_col]
                dist = (template - patch) ** 2
                ssd[i, j] = (dist * valid_mask).sum() / total_weight

        # Remove the case where sample == template
        ssd[i_max - offset, j_max - offset] = 1.
        i_match, j_match = np.transpose(np.where(ssd == ssd.min()))[0]

        if ssd[i_match, j_match] < max_thresh:
            image[i_max + t_row, j_max + t_col] += image[
                i_match + t_row, j_match + t_col] * mask_template
            confidence[i_max + t_row, j_max + t_col] += max_conf * mask_template
            mask[i_max + t_row, j_max + t_col] = 0
            progress = 1

        if progress == 0:
            max_thresh = 1.1 * max_thresh

    return image[offset:-offset, offset:-offset]


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
