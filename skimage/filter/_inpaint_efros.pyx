from __future__ import division
import numpy as np
from skimage.filter.rank import minimum
from skimage.morphology import disk

cimport numpy as cnp


cpdef _inpaint_efros(painted, mask, window, max_thresh):
    """This function performs constrained texture synthesis. It grows the
    texture of surrounding region into the unknown pixels. This implementation
    is pixel-based. Check the Notes Section for a brief overview of the
    algorithm.

    Parameters
    ----------
    painted : (M, N) array, uint8
        Input image whose texture is to be calculated.
    mask : (M, N) array, bool
        Texture for True values are to be synthesised.
    window : int
        Width of the neighborhood window. (window, window) patch about the
        pixel to be inpainted. Preferably odd, for symmetry.
    max_thresh : float
        Maximum tolerable SSD (Sum of Squared Difference) between the template
        around a pixel to be filled and an equal size image sample.

    Returns
    -------
    painted : array, float
        Texture synthesised image.

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

    For further information refer to [1]_.

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

    sigma = window / 3
    # Sigma definition is as in there pseudo code:
    # http://graphics.cs.cmu.edu/people/efros/research/NPS/alg.html
    gauss_mask = _gaussian(sigma, (window, window))

    while mask.any():
        # Generate the boundary of ROI (region to be synthesised)
        boundary = mask - minimum(mask, disk(1))
        if not boundary.any() and mask.any():
            # If the remaining region is 1-pixel thick
            boundary = mask

        bound_list = np.transpose(np.where(boundary == 1))

        for (i_b, j_b) in bound_list:
            template = painted[i_b + t_row, j_b + t_col]
            valid_mask = gauss_mask * (1 - mask[i_b + t_row, j_b + t_col])

            i_m, j_m = _sum_sq_diff(source_image[offset:-offset,
                                                 offset:-offset],
                                    mask[offset:-offset, offset:-offset],
                                    template, valid_mask, max_thresh, i_b, j_b)

            if i_m != -1 and j_m != -1:
                painted[i_b, j_b] = source_image[i_m + offset,
                                                 j_m + offset]
                mask[i_b, j_b] = False

    return painted[offset:-offset, offset:-offset]


cdef _sum_sq_diff(cnp.float_t[:, ::] image,
                  cnp.uint8_t[:, ::] mask,
                  cnp.float_t[:, ::] template,
                  cnp.float_t[:, ::] valid_mask,
                  cnp.float_t max_thresh,
                  Py_ssize_t i_b, Py_ssize_t j_b):
    """This function performs template matching. The metric used is Sum of
    Squared Difference (SSD). The input taken is the ``template`` who's match
    is to be found in image. See the section below on Notes.

    Parameters
    ----------
    image : array, float
        Initial unpadded input image of shape (M, N).
    mask : (M, N) array, bool
        Texture for True values are to be synthesised; unpadded.
    template : array, float
        (window, window) Template who's match is to be found in image.
    valid_mask : array, float
        (window, window), governs differences which are to be considered for
        SSD computation. Masks out the unknown or unfilled pixels and gives a
        higher weight to the center pixel, decreasing as the distance from
        center pixel increases.
    max_thresh : float
        Maximum tolerable SSD (Sum of Squared Difference) between the template
        around a pixel to be filled and an equal size image sample.
    i_b, j_b : int
        Template matching for this index value.

    Returns
    -------
    ssd : array, float
        (M - window + 1, N - window + 1) The desired SSD values for all
        positions in the image.

    Notes
    -----
    The valid samples from the image are those which completely lie in the
    known region of the image, i.e. not belonging to the padded boundary and
    not having any pixel from the region to be inpainted.

    """
    cdef:
        Py_ssize_t i, j, k, l, i_min = -1, j_min = -1
        cnp.float_t ssd, total_weight
        cnp.uint8_t window, offset, flag

    min_ssd = 1.
    window = template.shape[0]
    offset = (window // 2)
    total_weight = valid_mask.base.sum()
    for i in range(image.shape[0] - window + 1):
        for j in range(image.shape[1] - window + 1):
            if i == i_b - offset and j == j_b - offset:
                continue
            flag = 0
            for k in range(window):
                for l in range(window):
                    if mask[i + k, j + l]:
                        flag = 1
                        break
                if flag == 1:
                    break

            if flag == 1:
                continue
            ssd = 0
            for k in range(window):
                for l in range(window):
                    ssd += ((template[k, l] - image[i + k, j + l]) ** 2
                            * valid_mask[k, l])
            ssd /= total_weight
            if ssd < max_thresh:
                max_thresh = ssd
                i_min = i + offset
                j_min = j + offset

    return i_min, j_min


cdef _gaussian(sigma=0.5, size=None):
    """Gaussian kernel array with given sigma and shape about the center pixel.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation. (default: 0.5)
    size : tuple
        Shape of the output kernel.

    Returns
    -------
    gauss_mask : array, float
        Gaussian kernel of shape ``size``.

    """
    sigma2 = sigma ** 2

    x = np.arange(-(size[0] - 1) / 2.0, (size[0] - 1) / 2.0 + 0.1)
    y = np.arange(-(size[1] - 1) / 2.0, (size[1] - 1) / 2.0 + 0.1)

    Kx = np.exp(-x ** 2 / (2 * sigma2))
    Ky = np.exp(-y ** 2 / (2 * sigma2))
    gauss_mask = np.outer(Kx, Ky) / (2.0 * np.pi * sigma2)

    return gauss_mask / gauss_mask.sum()
