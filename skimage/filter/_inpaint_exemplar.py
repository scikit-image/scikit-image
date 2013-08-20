from __future__ import division
import numpy as np
from skimage.morphology import erosion, dilation, disk
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage.filters import gaussian_filter


def _inpaint_criminisi(painted, mask, window, max_thresh):
    """This function performs constrained synthesis. It grows the texture
    of surrounding region into the unknown pixels.

    Parameters
    ---------
    painted : (M, N) array, float
        Input image whose texture is to be calculated
    synth_mask : (M, N) array, bool
        Texture for True values are to be synthesised
    window : int
        Size of the neighborhood window

    Returns
    -------
    painted : (M, N) array, float
        Texture synthesised image

    References
    ---------
    .. [1] Criminisi, A., Pe ' ez, P., and Toyama, K. (2004). "Region filling
           and object removal by exemplar-based inpainting". IEEE Transactions
           on Image Processing, 13(9):1200-1212

    """

    source_image = painted.copy()
    offset = window // 2

    t_row, t_col = np.ogrid[-offset:offset + 1, -offset:offset + 1]
    sigma = window / 6.4
    gauss_mask = _gaussian(sigma, (window, window))
    confidence = 1. - mask

    while mask.any():
        # Generate the fill_front, boundary of ROI (region to be synthesised)
        fill_front = mask - erosion(mask, disk(1))
        if not fill_front.any() and mask.any():
            # If the remaining region is 1-pixel thick
            fill_front = mask

        # Generate the image gradient and normal vector to the boundary
        im_grad_y, im_grad_x = (np.gradient(gaussian_filter(painted, sigma=1))
                                * (1 - dilation(mask, disk(1))))
        im_grad_x = abs(im_grad_x)
        im_grad_y = abs(im_grad_y)
        mod_im_grad = np.hypot(im_grad_y, im_grad_x)
        ny, nx = np.gradient(mask.astype(np.float))

        # Generate the indices of the pixels in fill_front
        fill_front_indices = np.transpose(np.where(fill_front == 1))

        max_priority, max_conf, i_max, j_max = 0, 0, 0, 0

        # Determine the priority of pixels on the boundary, hence the order
        for k in range(fill_front_indices.shape[0]):
            i = fill_front_indices[k, 0]
            j = fill_front_indices[k, 1]

            # Compute the confidence term
            confidence_term = (confidence[i + t_row, j + t_col].sum() /
                               window ** 2)

            # Compute the data term
            mod_grad = mod_im_grad[i + t_row, j + t_col]
            ind_max = np.array([i, j], dtype=np.uint8)
            ind_max += np.transpose(np.where(mod_grad == mod_grad.max()))[0]
            ind_max = tuple(ind_max - offset)

            data_term = abs(im_grad_x[ind_max] * nx[i, j] +
                            im_grad_y[ind_max] * ny[i, j])
            data_term /= np.sqrt(mod_grad.max() *
                                (nx[i, j] ** 2 + ny[i, j] ** 2))

            # Compute the priority for determining the order for inpainting
            priority = data_term * confidence_term
            if priority > max_priority:
                max_priority = priority
                max_conf = confidence_term
                i_max, j_max = i, j

        template = painted[i_max + t_row, j_max + t_col]
        mask_template = mask[i_max + t_row, j_max + t_col]
        valid_mask = gauss_mask * (1 - mask_template)

        ssd = _sum_sq_diff(source_image, template, valid_mask)
        # Remove the case where sample == template
        ssd[i_max - offset, j_max - offset] = 1.

        i_match, j_match = np.transpose(np.where(ssd == ssd.min()))[0] + offset

        if ssd[i_match - offset, j_match - offset] < max_thresh:
            painted[i_max + t_row, j_max + t_col] += (painted[i_match + t_row,
                                                              j_match + t_col]
                                                      * mask_template)
            confidence[i_max + t_row, j_max + t_col] += (max_conf *
                                                         mask_template)
            mask[i_max + t_row, j_max + t_col] = 0
            progress = 1

        if progress == 0:
            max_thresh = 1.1 * max_thresh

    return painted[offset:-offset, offset:-offset]


def _sum_sq_diff(painted, template, valid_mask):
    """This function performs template matching. The metric used is Sum of
    Squared Difference (SSD). The input taken is the template who's match is
    to be found in image.

    Parameters
    ---------
    painted : array, float
        Input image of shape (M, N)
    template : array, float
        (window, window) Template who's match is to be found in painted.
    valid_mask : array, float
        (window, window), governs differences which are to be considered for
        SSD computation. Masks out the unknown or unfilled pixels and gives a
        higher weightage to the center pixel, decreasing as the distance from
        center pixel increases.

    Returns
    ------
    ssd : array, float
        (M - window +1, N - window + 1) The desired SSD values for all
        positions in the painted

    """
    total_weight = valid_mask.sum()
    window_size = template.shape
    y = as_strided(painted,
                   shape=(painted.shape[0] - window_size[0] + 1,
                          painted.shape[1] - window_size[1] + 1,) +
                   window_size,
                   strides=painted.strides * 2)
    ssd = np.einsum('ijkl, kl, kl->ij', y, template, valid_mask,
                    dtype=np.float)
    ssd *= - 2
    ssd += np.einsum('ijkl, ijkl, kl->ij', y, y, valid_mask)
    ssd += np.einsum('ij, ij, ij', template, template, valid_mask)
    return ssd / total_weight


def _gaussian(sigma=0.5, size=None):
    """Gaussian kernel array with given sigma and shape about the center pixel.

    Parameters
    ---------
    sigma : float
        Standard deviation
    size : tuple
        Shape of the output kernel

    Returns
    ------
    gauss_mask : array, float
        Gaussian kernel of shape ``size``

    """

    x = np.arange(-(size[0] - 1) / 2.0, (size[0] - 1) / 2.0 + 1)
    y = np.arange(-(size[1] - 1) / 2.0, (size[1] - 1) / 2.0 + 1)

    Kx = np.exp(-x ** 2 / (2 * sigma ** 2))
    Ky = np.exp(-y ** 2 / (2 * sigma ** 2))
    gauss_mask = np.outer(Kx, Ky) / (2.0 * np.pi * sigma ** 2)

    return gauss_mask / gauss_mask.sum()
