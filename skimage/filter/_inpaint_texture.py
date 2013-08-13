from __future__ import division
import numpy as np
from skimage.morphology import erosion, disk
from numpy.lib.stride_tricks import as_strided


def _inpaint_efros(painted, mask, window, max_thresh):
    """This function performs constrained texture synthesis. It grows the
    texture of surrounding region into the unknown pixels. This implementation
    is pixel-based. Check the Notes Section for a brief overview of the
    algorithm.

    Parameters
    ---------
    painted : (M, N) array, np.uint8
        Input image whose texture is to be calculated
    mask : (M, N) array, np.bool
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

    """

    source_image = painted.copy()
    offset = window // 2
    t_row, t_col = np.ogrid[-offset:offset + 1, -offset:offset + 1]

    sigma = window / 6.4
    gauss_mask = _gaussian(sigma, (window, window))

    while mask.any():
        progress = 0

        # Generate the boundary of ROI (region to be synthesised)
        boundary = mask - erosion(mask, disk(1))
        if not boundary.any():  # If the remaining region is 1-pixel thick
            boundary = mask

        bound_list = np.transpose(np.where(boundary == 1))

        for k in range(bound_list.shape[0]):
            i_b = bound_list[k, 0].astype(np.int16)
            j_b = bound_list[k, 1].astype(np.int16)
            template = painted[i_b + t_row, j_b + t_col]
            mask_template = mask[i_b + t_row, j_b + t_col]
            valid_mask = gauss_mask * (1 - mask_template)

            ssd = _sum_sq_diff(source_image, template, valid_mask)
            # Remove the case where `sample` == `template`
            ssd[i_b - offset, j_b - offset] = 1.

            matched_index = np.transpose(np.where(ssd == ssd.min()))[0]

            if ssd[tuple(matched_index)] < max_thresh:
                painted[i_b, j_b] = source_image[tuple(matched_index + offset)]
                mask[i_b, j_b] = False
                progress = 1

        if progress == 0:
            max_thresh = 1.1 * max_thresh

    return (painted[offset:-offset, offset:-offset])


def _sum_sq_diff(image, template, valid_mask):
    """This function performs template matching. The metric used is Sum of
    Squared Difference (SSD). The input taken is the template who's match is
    to be found in image.

    Parameters
    ---------
    image : array, np.float
        Input image of shape (M, N)
    template : array, np.float
        (window, window) Template who's match is to be found in image
    valid_mask : array, np.float
        (window, window), governs differences which are to be considered for
        SSD computation. Masks out the unknown or unfilled pixels and gives a
        higher weightage to the center pixel, decreasing as the distance from
        center pixel increases

    Returns
    ------
    ssd : array, np.float
        (M - window +1, N - window + 1) The desired SSD values for all
        positions in the image

    """
    total_weight = valid_mask.sum()
    window_size = template.shape
    y = as_strided(image,
                   shape=(image.shape[0] - window_size[0] + 1,
                          image.shape[1] - window_size[1] + 1,) +
                   window_size,
                   strides=image.strides * 2)
    # ``(y-template)**2`` followed by reduction -> 4D array intermediate
    # For einsum, labels are used to iterate through axes, order is imp: 'ij',
    # row wise iteration. Term after '->' represents the order for output array
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
    gauss_mask : array, np.float
        Gaussian kernel of shape ``size``

    """
    sigma = max(abs(sigma), 1e-10)

    x = np.arange(-(size[0] - 1) / 2.0, (size[0] - 1) / 2.0)
    y = np.arange(-(size[1] - 1) / 2.0, (size[1] - 1) / 2.0)

    Kx = np.exp(-x ** 2 / (2 * sigma ** 2))
    Ky = np.exp(-y ** 2 / (2 * sigma ** 2))
    gauss_mask = np.outer(Kx, Ky) / (2.0 * np.pi * sigma ** 2)

    return gauss_mask / gauss_mask.sum()
