import numpy as np
from skimage.morphology import erosion, disk
# from skimage.feature import match_template
from skimage.util import img_as_float


def growImage(input_image, synth_mask, window):
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

    """

    MAX_THRESH = 0.2

    # Padding
    pad_size = tuple(np.array(input_image.shape) + np.array(window) - 1)
    image = np.mean(input_image) * np.ones(pad_size, dtype=np.float32)
    mask = np.zeros(pad_size, bool)
    h, w = input_image.shape
    i0, j0 = window, window
    i0 /= 2
    j0 /= 2
    image[i0:i0 + h, j0:j0 + w] = img_as_float(input_image)
    mask[i0:i0 + h, j0:j0 + w] = synth_mask

    sigma = window / 6.4
    gauss_mask = _gaussian(sigma, (window, window))
    ssd = np.zeros(input_image.shape, np.float)

    while mask.any():
        progress = 0

        # Generate the boundary of ROI (region to be synthesised)
        boundary = mask - erosion(mask, disk(1))
        if not boundary.any():  # If the remaining region is 1-pixel thick
            boundary = mask

        bound_list = np.transpose(np.where(boundary == 1))

        for i_b, j_b in bound_list:
            template = image[(i_b - window / 2):(i_b + window / 2 + 1),
                             (j_b - window / 2):(j_b + window / 2 + 1)]
            mask_template = mask[(i_b - window / 2):(i_b + window / 2 + 1),
                                 (j_b - window / 2):(j_b + window / 2 + 1)]
            valid_mask = gauss_mask * (1 - mask_template)

            # best_matches = find_matches(template, valid_mask, image, window)
            total_weight = np.sum(valid_mask)
            for i in xrange(input_image.shape[0]):
                for j in xrange(input_image.shape[1]):
                    sample = image[i:i + window, j:j + window]
                    dist = (template - sample) ** 2
                    ssd[i, j] = np.sum(dist * valid_mask) / total_weight

            # Remove the case where sample == template
            ssd[i_b - window / 2, j_b - window / 2] = 1.

            best_matches = np.transpose(np.where(ssd == ssd.min()))

            matched_index = best_matches[0, :]

            if ssd[tuple(matched_index)] < MAX_THRESH:
                image[i_b, j_b] = image[tuple(matched_index + [window / 2,
                                                               window / 2])]
                mask[i_b, j_b] = False
                progress = 1

        if progress == 0:
            MAX_THRESH = 1.1 * MAX_THRESH

    return image[i0:-i0, j0:-j0]


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
