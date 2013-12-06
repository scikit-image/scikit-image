#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

from __future__ import division
import numpy as np
from skimage.morphology import disk
from skimage.filter.rank import minimum
from skimage.util import img_as_ubyte
from scipy.ndimage import gaussian_filter, sobel

cimport numpy as cnp
from libc.math cimport sqrt


cpdef _inpaint_criminisi(painted, mask, window, ssd_thresh):
    """This function performs constrained synthesis using Criminisi et al.
    algorithm. It grows the texture of the surrounding region to fill in
    unknown pixels. See Notes for an outline of the algorithm.

    Parameters
    ----------
    painted : (M, N) array, float
        Input image whose texture is to be calculated.
    mask : (M, N) array, int8
        Texture for True values are to be synthesised.
    window : int
        Width of the neighborhood window. ``(window, window)`` patch with
        centre at the pixel to be inpainted. Odd, for symmetry.
    ssd_thresh : float
        Maximum tolerable SSD (Sum of Squared Difference) between the template
        around a pixel to be filled and an equal size image sample for
        template matching.

    Returns
    -------
    painted : (M, N) array, float
        Final texture synthesised image.

    Notes
    -----
    For best results, ``window`` should be larger in size than the largest
    texel (texture element) being inpainted. A texel is the smallest repeating
    block of pixels in a texture or pattern. For example, in the case below of
    the ``skimage.data.checkerboard`` image, the single white/black square is
    the largest texel which is of shape ``(25, 25)``. A value larger than this
    yields perfect reconstruction, but in case of a value smaller than this
    perfect reconstruction may not be possible.

    Outline of the algorithm for Texture Synthesis is as follows:
    - Loop: Generate the boundary pixels of the region to be inpainted
        - Loop: Compute the priority of each pixel.
            - Generate a template of ``(window, window)``, center: boundary
              pixel.
            - confidence_term: avg amount of reliable information in template.
            - data_term: strength of the isophote hitting this boundary pixel.
            - ``priority = data_term * confidence_term``.
        - Repeat for all boundary pixels and chose the pixel with max priority.
        - Template matching of the pixel with max priority.
            - Generate a template of ``(window, window)`` around this pixel.
            - Compute the Sum of Squared Difference (SSD) between template and
              similar sized patches across the image.
            - Find the pixel with smallest SSD, such that patch isn't where
              template is located (False positive).
            - Update the intensity value of the unknown region of template as
              the corresponding value from matched patch.
    - Repeat until all pixels are inpainted.

    For further information refer to [1]_.

    References
    ----------
    .. [1] A. Criminisi, P. Perez, and K. Toyama. 2004. Region filling and
           object removal by exemplar-based image inpainting. Trans. Img. Proc.
           13, 9 (September 2004), 1200-1212. DOI=10.1109/TIP.2004.833105.

    """

    offset = window // 2
    inner = (slice(offset, -offset), slice(offset, -offset))
    source_image = painted[inner].copy()

    t_row, t_col = np.ogrid[-offset:offset + 1, -offset:offset + 1]
    sigma = window / 3
    gauss_mask = _gaussian(sigma, (window, window))
    confidence = 1. - mask
    mask_ubyte = mask.astype(np.uint8)
    mask_contig = np.ascontiguousarray(mask[inner])

    while True:
        # Generate the fill_front, boundary of ROI (region to be synthesised)
        fill_front = mask - minimum(mask_ubyte, disk(1))
        if not fill_front.any():
            if mask.any():
                # If the remaining region is 1-pixel thick
                fill_front = mask
            else:
                break

        smooth_painted = gaussian_filter(painted, sigma=1)
        smooth_painted[mask == 1] = np.nan

        # Generate the image gradient and normal vector to the boundary
        # Order reversed; image gradient rotated by 90 degree
        dx = sobel(smooth_painted, axis=0)
        dy = -sobel(smooth_painted, axis=1)
        dx[np.isnan(dx)] = 0
        dy[np.isnan(dy)] = 0
        smooth_painted[np.isnan(smooth_painted)] = 0
        ny = sobel(mask, axis=0)
        nx = sobel(mask, axis=1)

        # Priority calculation; pixel for which inpainting is done first
        i_max, j_max = _priority_calc(fill_front, confidence,
                                      dx, dy, nx, ny, window)

        template = painted[i_max + t_row, j_max + t_col]
        mask_template = mask[i_max + t_row, j_max + t_col]
        valid_mask = gauss_mask * (1 - mask_template)

        # Template matching
        i_m, j_m = _sum_sq_diff(source_image, mask_contig, template,
                                valid_mask, ssd_thresh, i_max, j_max)

        if i_m != -1 and j_m != -1:
            painted[i_max + t_row, j_max + t_col] += (mask_template *
                                                      painted[i_m + t_row,
                                                              j_m + t_col])
            confidence[i_max + t_row, j_max + t_col] += (mask_template *
                                                         confidence[i_max,
                                                                    j_max])
            mask_ubyte[i_max + t_row, j_max + t_col] = 0
            mask[i_max + t_row, j_max + t_col] = 0

    return painted[inner]


cdef _priority_calc(cnp.int16_t[:, ::1] fill_front,
                    cnp.float_t[:, ::1] confidence,
                    cnp.float_t[:, ::1] dx, cnp.float_t[:, ::1] dy,
                    cnp.int8_t[:, ::1] nx, cnp.int8_t[:, ::1] ny,
                    cnp.uint8_t window):
    """Calculation of the priority term for the region of interest boundary
    pixels to determine the order of filling.

    Parameters
    ----------
    fill_front : array, int8
        Boundary of the region to be inpainted.
    confidence : array, float
        Array containing the confidence value of the pixels.
    dx : array, float
        90 degree rotated gradient of the image in the X direction.
    dy : array, float
        90 degree rotated gradient of the image in the Y direction.
    nx : array, float
        X component of the unit vector at a pixel, perpendicular to the path of
        boundary of the region to be inpainted through this pixel.
    ny : array, float
        Y component of the unit vector at a pixel, perpendicular to the path of
        boundary of the region to be inpainted through this pixel.
    window : int
        Size of the neighborhood window. ``(window, window)`` patch with
        centre at the pixel to be inpainted.

    Returns
    -------
    i_max, j_max : tuple, int
        Index value of the pixel with the maximum priority value.

    """
    cdef:
        Py_ssize_t i, j, z, k, l, i_data, j_data, i_max = -1, j_max = -1
        Py_ssize_t[::1] fill_front_x, fill_front_y
        cnp.float_t max_priority = 0, conf, data_term, mod, max_mod, priority
        cnp.uint8_t offset

    offset = window // 2
    # Generate the indices of the pixels in fill_front
    fill_front_x, fill_front_y = np.ascontiguousarray(np.nonzero(fill_front))

    for z in range(fill_front_x.shape[0]):
        i = fill_front_x.base[z]
        j = fill_front_y.base[z]

        if fill_front_x.shape[0] == 1:
            # End case, when only 1 pixel is remaining to be inpainted
            return i, j

        # Computation of the data terms
        max_mod = 0
        i_data, j_data = 0, 0
        for k in range(-offset, offset + 1):
            for l in range(-offset, offset + 1):
                mod = dx[i + k, j + l] ** 2 + dy[i + k, j + l] ** 2
                if mod > max_mod:
                    max_mod = mod
                    i_data = i + k
                    j_data = j + l

        # If no variation in intensity (grad == 0), priority = 0
        if i_data == 0 and j_data == 0:
            continue
        if sqrt(nx[i, j] ** 2 + ny[i, j] ** 2) == 0:
            # Single pixel unpainted in the middle
            continue

        data_term = abs(dx[i_data, j_data] * nx[i, j] +
                        dy[i_data, j_data] * ny[i, j])
        data_term /= (sqrt(dx[i_data, j_data] ** 2 + dy[i_data, j_data] ** 2) *
                      sqrt(nx[i, j] ** 2 + ny[i, j] ** 2))

        # Compute the confidence terms
        conf = 0
        for k in range(-offset, offset + 1):
            for l in range(-offset, offset + 1):
                conf += confidence[i + k, j + l]
        conf /= (window ** 2)
        confidence[i, j] = conf

        priority = conf * data_term
        if priority > max_priority:
            max_priority = priority
            i_max = i
            j_max = j

    return i_max, j_max


cdef _sum_sq_diff(cnp.float_t[:, ::1] image,
                  cnp.int8_t[:, ::1] mask,
                  cnp.float_t[:, ::1] template,
                  cnp.float_t[:, ::1] valid_mask,
                  cnp.float_t ssd_thresh,
                  Py_ssize_t i_b, Py_ssize_t j_b):
    """This function performs template matching. The metric used is Sum of
    Squared Difference (SSD). The input taken is the ``template`` to be found
    in ``image``.

    Parameters
    ----------
    image : array, float
        Initial unpadded input image of shape (M, N).
    mask : (M, N) array, int8
        Texture for ``1`` values are to be synthesised.
    template : ``(window, window)`` array, float
        Template for which match is to be found in image.
    valid_mask : ``(window, window)`` array, float
        Masks out the unknown or unfilled pixels and gives a higher weight to
        the center pixel, decreasing as the distance from center pixel
        increases.
    ssd_thresh : float
        Maximum tolerable SSD (Sum of Squared Difference) between the template
        around a pixel to be filled and an equal size image sample for
        template matching.
    i_b, j_b : int
        Template matching is done for this index value.

    Returns
    -------
    ssd : ``(M - window + 1, N - window + 1)`` array, float
        The desired SSD values for all positions in the image.

    Notes
    -----
    The Valid samples from the image are those which completely lie in the
    known region of the image, i.e. not belonging to the padded boundary and
    not having any pixel from the region to be inpainted.

    """

    cdef:
        Py_ssize_t i, j, k, l, i_min = -1, j_min = -1
        cnp.float_t ssd, total_weight
        cnp.uint8_t window, offset, flag

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
                else:
                    continue
                break

            if flag == 1:
                continue
            ssd = 0
            for k in range(window):
                for l in range(window):
                    ssd += ((template[k, l] - image[i + k, j + l]) ** 2
                            * valid_mask[k, l])
            ssd /= total_weight
            if ssd < ssd_thresh:
                ssd_thresh = ssd
                i_min = i + offset
                j_min = j + offset

    return i_min + offset, j_min + offset


cdef _gaussian(sigma=0.5, size=None):
    """Gaussian kernel array with given sigma and shape about the center pixel.

    Parameters
    ----------
    sigma : float, optional
        Standard deviation (default: 0.5)
    size : tuple
        Shape of the output kernel.

    Returns
    -------
    gauss_mask : array, float
        Gaussian kernel of shape ``size``.

    """
    sigma2 = sigma ** 2
    x = np.arange(-(size[0] - 1) / 2.0, (size[0] - 1) / 2.0 + 1)
    y = np.arange(-(size[1] - 1) / 2.0, (size[1] - 1) / 2.0 + 1)

    Kx = np.exp(-x ** 2 / (2 * sigma2))
    Ky = np.exp(-y ** 2 / (2 * sigma2))
    gauss_mask = np.outer(Kx, Ky) / (2.0 * np.pi * sigma2)

    return gauss_mask / gauss_mask.sum()
