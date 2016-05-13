"""
Methods to characterize image textures.
"""

import numpy as np
from .._shared.utils import assert_nD
from ..util import img_as_float
from ..color import gray2rgb
from ._texture import (_glcm_loop,
                       _local_binary_pattern,
                       _multiblock_lbp)


def greycomatrix(image, distances, angles, levels=256, symmetric=False,
                 normed=False, clockwise=True):
    """Calculate the grey-level co-occurrence matrix.

    A grey level co-occurrence matrix is a histogram of co-occurring
    greyscale values at a given offset over an image.

    Parameters
    ----------
    image : array_like of uint8
        Integer typed input image. The image will be cast to uint8, so
        the maximum value must be less than 256.
    distances : array_like
        List of pixel pair distance offsets.
    angles : array_like
        List of pixel pair angles in radians.
    levels : int, optional
        The input image should contain integers in [0, levels-1],
        where levels indicate the number of grey-levels counted
        (typically 256 for an 8-bit image). The maximum value is
        256.
    symmetric : bool, optional
        If True, the output matrix `P[:, :, d, theta]` is symmetric. This
        is accomplished by ignoring the order of value pairs, so both
        (i, j) and (j, i) are accumulated when (i, j) is encountered
        for a given offset. The default is False.
    normed : bool, optional
        If True, normalize each matrix `P[:, :, d, theta]` by dividing
        by the total number of accumulated co-occurrences for the given
        offset. The elements of the resulting matrix sum to 1. The
        default is False.
    clockwise : bool, optional
        Defines if the angles will be considered clockwise or anti-clockwise. Default: True

    Returns
    -------
    P : 4-D ndarray
        The grey-level co-occurrence histogram. The value
        `P[i,j,d,theta]` is the number of times that grey-level `j`
        occurs at a distance `d` and at an angle `theta` from
        grey-level `i`. If `normed` is `False`, the output is of
        type uint32, otherwise it is float64.

    References
    ----------
    .. [1] The GLCM Tutorial Home Page,
           http://www.fp.ucalgary.ca/mhallbey/tutorial.htm
    .. [2] Pattern Recognition Engineering, Morton Nadler & Eric P.
           Smith
    .. [3] Wikipedia, http://en.wikipedia.org/wiki/Co-occurrence_matrix


    Examples
    --------
    Compute 2 GLCMs: One for a 1-pixel offset to the right, and one
    for a 1-pixel offset upwards.

    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> result = greycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=4)
    >>> result[:, :, 0, 0]
    array([[2, 2, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 1],
           [0, 0, 0, 1]], dtype=uint32)
    >>> result[:, :, 0, 1]
    array([[1, 1, 3, 0],
           [0, 1, 1, 0],
           [0, 0, 0, 2],
           [0, 0, 0, 0]], dtype=uint32)
    >>> result[:, :, 0, 2]
    array([[3, 0, 2, 0],
           [0, 2, 2, 0],
           [0, 0, 1, 2],
           [0, 0, 0, 0]], dtype=uint32)
    >>> result[:, :, 0, 3]
    array([[2, 0, 0, 0],
           [1, 1, 2, 0],
           [0, 0, 2, 1],
           [0, 0, 0, 0]], dtype=uint32)

    """
    assert_nD(image, 2)
    assert_nD(distances, 1, 'distances')
    assert_nD(angles, 1, 'angles')

    assert levels <= 256
    image = np.ascontiguousarray(image)
    assert image.min() >= 0
    assert image.max() < levels
    image = image.astype(np.uint8)
    distances = np.ascontiguousarray(distances, dtype=np.float64)
    angles = np.ascontiguousarray(angles, dtype=np.float64)

    P = np.zeros((levels, levels, len(distances), len(angles)),
                 dtype=np.uint32, order='C')

    # count co-occurences
    _glcm_loop(image, distances, angles, levels, P, clockwise)

    # make each GLMC symmetric
    if symmetric:
        Pt = np.transpose(P, (1, 0, 2, 3))
        P = P + Pt

    # normalize each GLMC
    if normed:
        P = P.astype(np.float64)
        glcm_sums = np.apply_over_axes(np.sum, P, axes=(0, 1))
        glcm_sums[glcm_sums == 0] = 1
        P /= glcm_sums

    return P


def greycoprops(P, prop='contrast'):
    """Calculate texture properties of a GLCM.

    Compute a feature of a grey level co-occurrence matrix to serve as
    a compact summary of the matrix. The properties are computed as
    follows:

    - 'ASM': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}^2`
    - 'autocorr': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(ij)`
    - 'contrast': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i-j)^2`
    - 'correlation':
        .. math:: \\sum_{i,j=0}^{levels-1} P_{i,j}\\left[\\frac{(i-\\mu_i) \\
                  (j-\\mu_j)}{\\sqrt{(\\sigma_i^2)(\\sigma_j^2)}}\\right]        
    - 'cprominence': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i+j-\\mu_i-\\mu_j)^4`
    - 'cshade': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i+j-\\mu_i-\\mu_j)^3`
    - 'dissimilarity': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}|i-j|`
    - 'energy': :math:`\\sqrt{ASM}`
    - 'entropy': :math:`\\sum_{i,j=0}^{levels-1} -P_{i,j}*\\log{P_{i,j}}`
    - 'homogeneity': :math:`\\sum_{i,j=0}^{levels-1}\\frac{P_{i,j}}{1+(i-j)^2}`
    - 'invdiff': :math:`\\sum_{i,j=0}^{levels-1} \\frac{P_{i,j}}{1+|i-j|}`
    - 'maxprob': :math:`\\sum_{i,j=0}^{levels-1} \\max{P_{i,j}}`
    - 'mean': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i)`
    - 'variance': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i-\\mu_i)^2`

    Parameters
    ----------
    P : ndarray
        Input array. `P` is the grey-level co-occurrence histogram
        for which to compute the specified property. The value
        `P[i,j,d,theta]` is the number of times that grey-level j
        occurs at a distance d and at an angle theta from
        grey-level i.
    prop : {'ASM', 'autocorr', 'contrast', 'correlation', 'cprominence', 'cshade', 'dissimilarity', 'energy', 'entropy', 'homogeneity', \
            'invdiff', 'maxprob', 'mean', 'variance'}, optional
        The property of the GLCM to compute. The default is 'contrast'.

    Returns
    -------
    results : 2-D ndarray
        2-dimensional array. `results[d, a]` is the property 'prop' for
        the d'th distance and the a'th angle.

    References
    ----------
    .. [1] The GLCM Tutorial Home Page,
           http://www.fp.ucalgary.ca/mhallbey/tutorial.htm
           
    .. [2] R. M. Haralick, K. Shanmugan, and I. H. Dinstein,
        "Textural features for image classification,"
        IEEE Trans. Syst., Man, Cybern., vol. SMC-3, pp. 610–621, May 1973.

    .. [3] R. W. Conners, M. M. Trivedi, and C. A. Harlow,
        "Segmentation of a high-resolution urban scene using texture operators,"
        Comput. Vision, Graph., Image Processing, vol. 25, pp. 273–310, 1984.

    .. [4] R. M. Haralick, "Statistical and structural approaches to texture,"
        Proc. IEEE, vol. 67, pp. 786–804, May 1979.

    .. [5] Soh, L.-K.; Tsatsoulis, C.,
        "Texture analysis of SAR sea ice imagery using gray level co-occurrence matrices,"
        in Geoscience and Remote Sensing, IEEE Transactions on , vol.37, no.2, pp.780-795, Mar 1999

    Examples
    --------
    Compute the contrast for GLCMs with distances [1, 2] and angles
    [0 degrees, 90 degrees]

    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> g = greycomatrix(image, [1, 2], [0, np.pi/2], levels=4,
    ...                  normed=True, symmetric=True)
    >>> contrast = greycoprops(g, 'contrast')
    >>> contrast
    array([[ 0.58333333,  1.        ],
           [ 1.25      ,  2.75      ]])

    """
    assert_nD(P, 4, 'P')

    (num_level, num_level2, num_dist, num_angle) = P.shape
    assert num_level == num_level2
    assert num_dist > 0
    assert num_angle > 0

    # create weights for specified property
    I, J = np.ogrid[0:num_level, 0:num_level]
    if prop == 'contrast':
        weights = (I - J) ** 2
    elif prop == 'dissimilarity':
        weights = np.abs(I - J)
    elif prop == 'homogeneity':
        weights = 1. / (1. + (I - J) ** 2)
    elif prop == 'invdiff':
        weights = 1. / (1. + np.abs(I - J))
    elif prop == 'autocorr':
        weights = I * J
    elif prop in ['ASM', 'energy', 'correlation', 'cshade', 'cprominence', 'mean', 'entropy', 'variance', 'maxprob']:
        pass
    else:
        raise ValueError('%s is an invalid property' % (prop))

    # compute property for each GLCM
    if prop == 'energy':
        asm = np.apply_over_axes(np.sum, (P ** 2), axes=(0, 1))[0, 0]
        results = np.sqrt(asm)
    elif prop == 'ASM':
        results = np.apply_over_axes(np.sum, (P ** 2), axes=(0, 1))[0, 0]
    elif prop == 'entropy':
        results = np.apply_over_axes(np.sum, -P * np.log(P + np.finfo(np.float).eps), axes=(0, 1))[0, 0]
    elif prop == 'mean':
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        results = np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
    elif prop == 'variance':
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        mean_i = np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
        weights = np.power(I - mean_i, 2)
        results = np.apply_over_axes(np.sum, (P * weights), axes=(0, 1))[0, 0]
    elif prop == 'maxprob':
        results = np.apply_over_axes(np.max, P, axes=(0, 1))[0, 0]
    elif prop == 'correlation':
        results = np.zeros((num_dist, num_angle), dtype=np.float64)
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))
        diff_i = I - np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
        diff_j = J - np.apply_over_axes(np.sum, (J * P), axes=(0, 1))[0, 0]

        std_i = np.sqrt(np.apply_over_axes(np.sum, (P * (diff_i) ** 2),
                                           axes=(0, 1))[0, 0])
        std_j = np.sqrt(np.apply_over_axes(np.sum, (P * (diff_j) ** 2),
                                           axes=(0, 1))[0, 0])
        cov = np.apply_over_axes(np.sum, (P * (diff_i * diff_j)),
                                 axes=(0, 1))[0, 0]

        # handle the special case of standard deviations near zero
        mask_0 = std_i < 1e-15
        mask_0[std_j < 1e-15] = True
        results[mask_0] = 1

        # handle the standard case
        mask_1 = mask_0 == False
        results[mask_1] = cov[mask_1] / (std_i[mask_1] * std_j[mask_1])
    elif prop == 'cshade':
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))

        mean_i = np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
        mean_j = np.apply_over_axes(np.sum, (J * P), axes=(0, 1))[0, 0]

        weights = np.power(I + J - mean_i - mean_j, 3)

        results = np.apply_over_axes(np.sum, (P * weights), axes=(0, 1))[0, 0]
    elif prop == 'cprominence':
        I = np.array(range(num_level)).reshape((num_level, 1, 1, 1))
        J = np.array(range(num_level)).reshape((1, num_level, 1, 1))

        mean_i = np.apply_over_axes(np.sum, (I * P), axes=(0, 1))[0, 0]
        mean_j = np.apply_over_axes(np.sum, (J * P), axes=(0, 1))[0, 0]

        weights = np.power(I + J - mean_i - mean_j, 4)

        results = np.apply_over_axes(np.sum, (P * weights), axes=(0, 1))[0, 0]
    elif prop in ['contrast', 'dissimilarity', 'homogeneity', 'invdiff', 'autocorr']:
        weights = weights.reshape((num_level, num_level, 1, 1))
        results = np.apply_over_axes(np.sum, (P * weights), axes=(0, 1))[0, 0]

    return results


def local_binary_pattern(image, P, R, method='default'):
    """Gray scale and rotation invariant LBP (Local Binary Patterns).

    LBP is an invariant descriptor that can be used for texture classification.

    Parameters
    ----------
    image : (N, M) array
        Graylevel image.
    P : int
        Number of circularly symmetric neighbour set points (quantization of
        the angular space).
    R : float
        Radius of circle (spatial resolution of the operator).
    method : {'default', 'ror', 'uniform', 'var'}
        Method to determine the pattern.

        * 'default': original local binary pattern which is gray scale but not
            rotation invariant.
        * 'ror': extension of default implementation which is gray scale and
            rotation invariant.
        * 'uniform': improved rotation invariance with uniform patterns and
            finer quantization of the angular space which is gray scale and
            rotation invariant.
        * 'nri_uniform': non rotation-invariant uniform patterns variant
            which is only gray scale invariant [2]_.
        * 'var': rotation invariant variance measures of the contrast of local
            image texture which is rotation but not gray scale invariant.

    Returns
    -------
    output : (N, M) array
        LBP image.

    References
    ----------
    .. [1] Multiresolution Gray-Scale and Rotation Invariant Texture
           Classification with Local Binary Patterns.
           Timo Ojala, Matti Pietikainen, Topi Maenpaa.
           http://www.rafbis.it/biplab15/images/stories/docenti/Danielriccio/Articoliriferimento/LBP.pdf, 2002.
    .. [2] Face recognition with local binary patterns.
           Timo Ahonen, Abdenour Hadid, Matti Pietikainen,
           http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.214.6851,
           2004.
    """
    assert_nD(image, 2)

    methods = {
        'default': ord('D'),
        'ror': ord('R'),
        'uniform': ord('U'),
        'nri_uniform': ord('N'),
        'var': ord('V')
    }
    image = np.ascontiguousarray(image, dtype=np.double)
    output = _local_binary_pattern(image, P, R, methods[method.lower()])
    return output


def multiblock_lbp(int_image, r, c, width, height):
    """Multi-block local binary pattern (MB-LBP).

    The features are calculated similarly to local binary patterns (LBPs),
    (See :py:meth:`local_binary_pattern`) except that summed blocks are
    used instead of individual pixel values.

    MB-LBP is an extension of LBP that can be computed on multiple scales
    in constant time using the integral image. Nine equally-sized rectangles
    are used to compute a feature. For each rectangle, the sum of the pixel
    intensities is computed. Comparisons of these sums to that of the central
    rectangle determine the feature, similarly to LBP.

    Parameters
    ----------
    int_image : (N, M) array
        Integral image.
    r : int
        Row-coordinate of top left corner of a rectangle containing feature.
    c : int
        Column-coordinate of top left corner of a rectangle containing feature.
    width : int
        Width of one of the 9 equal rectangles that will be used to compute
        a feature.
    height : int
        Height of one of the 9 equal rectangles that will be used to compute
        a feature.

    Returns
    -------
    output : int
        8-bit MB-LBP feature descriptor.

    References
    ----------
    .. [1] Face Detection Based on Multi-Block LBP
           Representation. Lun Zhang, Rufeng Chu, Shiming Xiang, Shengcai Liao,
           Stan Z. Li
           http://www.cbsr.ia.ac.cn/users/scliao/papers/Zhang-ICB07-MBLBP.pdf
    """

    int_image = np.ascontiguousarray(int_image, dtype=np.float32)
    lbp_code = _multiblock_lbp(int_image, r, c, width, height)
    return lbp_code


def draw_multiblock_lbp(img, r, c, width, height,
                        lbp_code=0,
                        color_greater_block=[1, 1, 1],
                        color_less_block=[0, 0.69, 0.96],
                        alpha=0.5
                        ):
    """Multi-block local binary pattern visualization.

    Blocks with higher sums are colored with alpha-blended white rectangles,
    whereas blocks with lower sums are colored alpha-blended cyan. Colors
    and the `alpha` parameter can be changed.

    Parameters
    ----------
    img : ndarray of float or uint
        Image on which to visualize the pattern.
    r : int
        Row-coordinate of top left corner of a rectangle containing feature.
    c : int
        Column-coordinate of top left corner of a rectangle containing feature.
    width : int
        Width of one of 9 equal rectangles that will be used to compute
        a feature.
    height : int
        Height of one of 9 equal rectangles that will be used to compute
        a feature.
    lbp_code : int
        The descriptor of feature to visualize. If not provided, the
        descriptor with 0 value will be used.
    color_greater_block : list of 3 floats
        Floats specifying the color for the block that has greater
        intensity value. They should be in the range [0, 1].
        Corresponding values define (R, G, B) values. Default value
        is white [1, 1, 1].
    color_greater_block : list of 3 floats
        Floats specifying the color for the block that has greater intensity
        value. They should be in the range [0, 1]. Corresponding values define
        (R, G, B) values. Default value is cyan [0, 0.69, 0.96].
    alpha : float
        Value in the range [0, 1] that specifies opacity of visualization.
        1 - fully transparent, 0 - opaque.

    Returns
    -------
    output : ndarray of float
        Image with MB-LBP visualization.

    References
    ----------
    .. [1] Face Detection Based on Multi-Block LBP
           Representation. Lun Zhang, Rufeng Chu, Shiming Xiang, Shengcai Liao,
           Stan Z. Li
           http://www.cbsr.ia.ac.cn/users/scliao/papers/Zhang-ICB07-MBLBP.pdf
    """

    # Default colors for regions.
    # White is for the blocks that are brighter.
    # Cyan is for the blocks that has less intensity.
    color_greater_block = np.asarray(color_greater_block, dtype=np.float64)
    color_less_block = np.asarray(color_less_block, dtype=np.float64)

    # Copy array to avoid the changes to the original one.
    output = np.copy(img)

    # As the visualization uses RGB color we need 3 bands.
    if len(img.shape) < 3:
        output = gray2rgb(img)

    # Colors are specified in floats.
    output = img_as_float(output)

    # Offsets of neighbour rectangles relative to central one.
    # It has order starting from top left and going clockwise.
    neighbour_rect_offsets = ((-1, -1), (-1, 0), (-1, 1),
                              (0, 1), (1, 1), (1, 0),
                              (1, -1), (0, -1))

    # Pre-multiply the offsets with width and height.
    neighbour_rect_offsets = np.array(neighbour_rect_offsets)
    neighbour_rect_offsets[:, 0] *= height
    neighbour_rect_offsets[:, 1] *= width

    # Top-left coordinates of central rectangle.
    central_rect_r = r + height
    central_rect_c = c + width

    for element_num, offset in enumerate(neighbour_rect_offsets):

        offset_r, offset_c = offset

        curr_r = central_rect_r + offset_r
        curr_c = central_rect_c + offset_c

        has_greater_value = lbp_code & (1 << (7-element_num))

        # Mix-in the visualization colors.
        if has_greater_value:
            new_value = ((1-alpha) * output[curr_r:curr_r+height, curr_c:curr_c+width]
                         + alpha * color_greater_block)
            output[curr_r:curr_r+height, curr_c:curr_c+width] = new_value
        else:
            new_value = ((1-alpha) * output[curr_r:curr_r+height, curr_c:curr_c+width]
                         + alpha * color_less_block)
            output[curr_r:curr_r+height, curr_c:curr_c+width] = new_value

    return output
