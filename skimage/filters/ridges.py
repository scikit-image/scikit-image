"""
Ridge filters.

Ridge filters can be used to detect continuous edges, such as vessels,
neurites, wrinkles, rivers, and other tube-like structures. The present
class of ridge filters relies on the eigenvalues of the Hessian matrix of
image intensities to detect tube-like structures where the intensity changes
perpendicular but not along the structure.
"""

from itertools import combinations_with_replacement

from warnings import warn

import numpy as np

from ..util import img_as_float, invert
from ..feature import hessian_matrix, hessian_matrix_eigvals
from .._shared.utils import assert_nD


def divide_nonzero(array1, array2, cval=1e-10):
    """
    Divides two arrays.

    Denominator is set to small value where zero to avoid ZeroDivisionError and
    return finite float array.

    Parameters
    ----------
    array1 : (N, ..., M) ndarray
        Array 1 in the enumerator.
    array2 : (N, ..., M) ndarray
        Array 2 in the denominator.
    cval : float, optional
        Value used to replace zero entries in the denominator.

    Returns
    -------
    array : (N, ..., M) ndarray
        Quotient of the array division.
    """

    # Copy denominator
    denominator = np.copy(array2)

    # Set zero entries of denominator to small value
    denominator[denominator == 0] = cval

    # Return quotient
    return np.divide(array1, denominator)


def hessian_nd_matrix(hessian_elements, ndim, order='rc'):
    """
    Generate fell Hessian matrices from Hessian elements of n-dimensional
    image.

    Parameters
    ----------
    hessian_elements : (E, N, ..., M) ndarray
        Array with Hessian elements for each image pixel.
    ndim : int
        Dimensions of input image.
     order : {'xy', 'rc'}, optional
        This parameter allows for the use of reverse or forward order of
        the image axes in gradient computation. 'xy' indicates the usage
        of the last axis initially (Hxx, Hxy, Hyy), whilst 'rc' indicates
        the use of the first axis initially (Hrr, Hrc, Hcc).

    Returns
    -------
    full : (n, n, N, ..., M) ndarray
        Array with full Hessian matrices for each image pixel.
    """

    # Generate empty array for storing Hessian matrices for each pixel
    d_hessian = (ndim, ndim)
    d_image = hessian_elements[0].shape
    full = np.zeros(d_hessian + d_image)

    # Generate list of image dimensions
    axes = range(ndim)
    if order == 'rc':
        axes = reversed(axes)

    # Fill Hessian matrices with Hessian elements
    for index, (ax0, ax1) in enumerate(combinations_with_replacement(axes, 2)):
        element = hessian_elements[index]
        full[ax0, ax1] = element
        if ax0 != ax1:
            full[ax1, ax0] = element

    # Reshape array such that Hessian matrices are given by the last indices
    d_hessian = list(range(2))
    d_image = list(range(2, 2 + ndim))
    full = np.transpose(full, d_image + d_hessian)

    # Return array with full Hessian matrices
    return full


def hessian_nd_eigenvalues(hessian, ndim):
    """
    Eigenvalues of Hessian matrices of n-dimensional image.

    Parameters
    ----------
    hessian : (n, n, M_1, ..., M_ndim) ndarray
        Array with n-dimensional Hessian matrices for each image pixel.
    ndim : int
        Dimensions of input image.

    Returns
    -------
    eigenvalues : (n, M_1, ..., M_ndim) ndarray
        Array with n Hessian eigenvalues for each image pixel.
    """

    # Compute Hessian eigenvalues
    eigenvalues = np.linalg.eigvalsh(hessian)

    # Reshape array such that eigenvalues are given by the first index
    d_image = list(range(ndim))
    d_eigen = list(range(ndim, ndim + 1))
    eigenvalues = np.transpose(eigenvalues, d_eigen + d_image)

    # Return array with Hessian eigenvalues
    return eigenvalues


def sortbyabs(array, axis=0):
    """
    Sort array along a given axis by absolute values.

    Parameters
    ----------
    array : (N, ..., M) ndarray
        Array with input image data.
    axis : int
        Axis along which to sort.

    Returns
    -------
    array : (N, ..., M) ndarray
        Array sorted along a given axis by absolute values.

    Notes
    -----
    Modified from: http://stackoverflow.com/a/11253931/4067734
    """

    # Create auxiliary array for indexing
    index = list(np.ix_(*[np.arange(i) for i in array.shape]))

    # Get indices of abs sorted array
    index[axis] = np.abs(array).argsort(axis)

    # Return abs sorted array
    return array[index]


def _compute_hessian_eigenvalues(image, sigma, sorting='none'):
    """
    Compute Hessian eigenvalues of nD images.

    For 2D images, the computation uses a more efficient, skimage-based
    algorithm.

    Parameters
    ----------
    image : (N, ..., M) ndarray
        Array with input image data.
    sigma : float
        Smoothing factor of image for detection of structures at different
        (sigma) scales.
    sorting : {'val', 'abs', 'none'}, optional
        Sorting of eigenvalues by values ('val') or absolute values ('abs'),
        or without sorting ('none'). Default is 'none'.

    Returns
    -------
    eigenvalues : (D, N, ..., M) ndarray
        Array with (sorted) eigenvalues of Hessian eigenvalues for each pixel
        of the input image.
    """

    # Get image dimensions
    ndim = image.ndim

    # Convert image to float
    image = img_as_float(image)

    # Make nD hessian
    elements = hessian_matrix(image, sigma=sigma, order='rc')

    # Correct for scale
    elements = [(sigma ** 2) * e for e in elements]

    if ndim == 2:

        # Compute 2D Hessian eigenvalues
        eigenvalues = np.array(hessian_matrix_eigvals(*elements))

    elif ndim > 2:

        # Make nD hessian
        hessian = hessian_nd_matrix(elements, ndim, order='rc')

        # Compute nD Hessian eigenvalues
        eigenvalues = hessian_nd_eigenvalues(hessian, ndim)

    else:

        # Check image dimensions
        assert_nD(image, [2, 'more'])

    if sorting == 'abs':

        # Sort eigenvalues by absolute values in ascending order
        eigenvalues = sortbyabs(eigenvalues, axis=0)

    elif sorting == 'val':

        # Sort eigenvalues by values in ascending order
        eigenvalues = np.sort(eigenvalues, axis=0)

    # Return Hessian eigenvalues
    return eigenvalues


def meijering(image, scale_range=(1, 10), scale_step=2, alpha=None,
              black_ridges=True):
    """
    Filter an image with the Meijering neuriteness filter.

    This filter can be used to detect continuous ridges, e.g. neurites,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Calculates the eigenvectors of the Hessian to compute the similarity of
    an image region to neurites, according to the method described in _[1].

    Parameters
    ----------
    image : (N, M) ndarray
        Array with input image data.
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    alpha : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a plate-like structure.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.

    Returns
    -------
    out : (N, M) ndarray
        Filtered image (maximum of pixels across all scales).

    References
    ----------
    .. [1] Meijering, E., Jacob, M., Sarria, J. C., Steiner, P., Hirling, H.,
        Unser, M. (2004). Design and validation of a tool for neurite tracing
        and analysis in fluorescence microscopy images. Cytometry Part A,
        58(2), 167-176.
    """

    # Check (sigma) scales
    sigmas = np.arange(scale_range[0], scale_range[1], scale_step)
    if np.any(np.asarray(sigmas) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")

    # Get image dimensions
    ndim = image.ndim

    # Set parameters
    if alpha is None:
        alpha = 1.0 / ndim

    # Invert image to detect bright ridges on dark background
    if not black_ridges:
        image = invert(image)

    # Generate empty (n+1)D arrays for storing auxiliary images filtered at
    # different (sigma) scales
    filtered_array = np.zeros(sigmas.shape + image.shape)

    # Filtering for all (sigma) scales
    for i, sigma in enumerate(sigmas):

        # Calculate (sorted) eigenvalues
        eigenvalues = _compute_hessian_eigenvalues(image, sigma, sorting='val')

        if ndim > 1:

            # Set coefficients for scaling eigenvalues
            coefficients = [alpha] * ndim
            coefficients[0] = 1

            # Compute auxiliary variables l_i = e_i + sum_{j!=i} alpha * e_j
            auxiliary = [np.sum([eigenvalues[i] * np.roll(coefficients, j)[i]
                         for j in range(ndim)], axis=0) for i in range(ndim)]

            # Compute maximum over auxiliary variables
            auxiliary = np.max(auxiliary, axis=0)

            # Rescale image intensity
            filtered = np.abs(auxiliary) / np.abs(np.max(auxiliary))

            # Remove background
            filtered = np.where(filtered > 0, filtered, 0)

            # Store results in (n+1)D matrices
            filtered_array[i] = filtered

        else:

            # Check image dimensions
            assert_nD(image, [2, 'more'])

    # Return for every pixel the value of the (sigma) scale with the maximum
    # output pixel value
    return np.max(filtered_array, axis=0)


def sato(image, scale_range=(1, 10), scale_step=2, black_ridges=True):
    """
    Filter an image with the Sato tubeness filter.

    This filter can be used to detect continuous ridges, e.g. tubes,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Calculates the eigenvectors of the Hessian to compute the similarity of
    an image region to tubes, according to the method described in _[1].

    Parameters
    ----------
    image : (N, M) ndarray
        Array with input image data.
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    alpha : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a plate-like structure.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.

    Returns
    -------
    out : (N, M) ndarray
        Filtered image (maximum of pixels across all scales).

    References
    ----------
    .. [1] Sato, Y., Nakajima, S., Shiraga, N., Atsumi, H., Yoshida, S.,
        Koller, T., ..., Kikinis, R. (1998). Three-dimensional multi-scale line
        filter for segmentation and visualization of curvilinear structures in
        medical images. Medical image analysis, 2(2), 143-168.
    """

    # Check (sigma) scales
    sigmas = np.arange(scale_range[0], scale_range[1], scale_step)
    if np.any(np.asarray(sigmas) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")

    # Get image dimensions
    ndim = image.ndim

    # Invert image to detect bright ridges on dark background
    if not black_ridges:
        image = invert(image)

    # Generate empty (n+1)D arrays for storing auxiliary images filtered
    # at different (sigma) scales
    filtered_array = np.zeros(sigmas.shape + image.shape)

    # Filtering for all (sigma) scales
    for i, sigma in enumerate(sigmas):

        # Calculate (sorted) eigenvalues
        eigenvalues = _compute_hessian_eigenvalues(image, sigma, sorting='val')

        if ndim == 2:

            # Get Hessian eigenvalues
            (lambda1, lambda2) = eigenvalues

            # Compute tubeness
            filtered = np.abs(lambda2)

            # Remove background
            filtered = np.where(lambda2 > 0, filtered, 0)

            # Store results in (n+1)D matrices
            filtered_array[i] = filtered

        elif ndim == 3:

            # Get Hessian eigenvalues
            (lambda1, lambda2, lambda3) = eigenvalues

            # Compute filtered image
            filtered = np.sqrt(np.abs(lambda2 * lambda3))

            # Remove background
            filtered = np.where(lambda3 > 0, filtered, 0)

            # Store results in (n+1)D matrices
            filtered_array[i] = filtered

        else:

            # Check image dimensions
            assert_nD(image, [2, 3])

    # Return for every pixel the value of the (sigma) scale with the maximum
    # output pixel value
    return np.max(filtered_array, axis=0)


def frangi(image, scale_range=(1, 10), scale_step=2, beta1=None, beta2=None,
           alpha=0.5, beta=0.5, gamma=15, black_ridges=True):
    """
    Filter an image with the Frangi vesselness filter.

    This filter can be used to detect continuous ridges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Calculates the eigenvectors of the Hessian to compute the similarity of
    an image region to vessels, according to the method described in _[1].

    Parameters
    ----------
    image : (N, M) ndarray
        Array with input image data.
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    alpha : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a plate-like structure.
    beta = beta1 : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a blob-like structure.
    gamma = beta2 : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to areas of high variance/texture/structure.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.

    Returns
    -------
    out : (N, M) ndarray
        Filtered image (maximum of pixels across all scales).

    Notes
    -----
    Written by Marc Schrijver, November 2001
    Re-Written by D. J. Kroon University of Twente, May 2009, _[2]

    References
    ----------
    .. [1] Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A.
        (1998,). Multiscale vessel enhancement filtering. In International
        Conference on Medical Image Computing and Computer-Assisted
        Intervention (pp. 130-137). Springer Berlin Heidelberg.
    .. [2] Kroon, D.J.: Hessian based Frangi vesselness filter.
    """

    # Check deprecated keyword parameters
    if beta1:
        warn("""Use keyword parameter 'beta' instead of 'beta1' which
                will be removed in version 0.16.""")
        beta = beta1

    if beta1:
        warn("""Use keyword parameter 'gamma' instead of 'beta2' which
                will be removed in version 0.16.""")
        gamma = beta2

    # Check (sigma) scales
    sigmas = np.arange(scale_range[0], scale_range[1], scale_step)
    if np.any(np.asarray(sigmas) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")

    # Rescale filter parameters
    alpha = 2 * alpha ** 2
    beta = 2 * beta ** 2
    gamma = 2 * gamma ** 2

    # Get image dimensions
    ndim = image.ndim
    shape = image.shape

    # Invert image to detect dark ridges on light background
    if black_ridges:
        image = invert(image)

    # Generate empty (n+1)D arrays for storing auxiliary images filtered
    # at different (sigma) scales
    filtered_array = np.zeros(sigmas.shape + image.shape)
    lambdas_array = np.zeros(sigmas.shape + image.shape)

    # Filtering for all (sigma) scales
    for i, sigma in enumerate(sigmas):

        # Calculate (abs sorted) eigenvalues
        eigenvalues = _compute_hessian_eigenvalues(image, sigma, sorting='abs')

        if ndim == 2:

            # Get Hessian eigenvalues
            (lambda1, lambda2) = eigenvalues

            # Compute sensitivity to deviation from a blob-like structure
            r_b = divide_nonzero(lambda1, lambda2) ** 2

            # Compute sensitivity to areas of high variance/texture/structure
            r_g = lambda1 ** 2 + lambda2 ** 2

            # Compute output image for given (sigma) scale
            filtered = (np.exp(-r_b / beta) *
                        (np.ones(shape) - np.exp(-r_g / gamma)))

            # Store results in (2+1)D matrices
            filtered_array[i] = filtered
            lambdas_array[i] = lambda2

        elif ndim == 3:

            # Get Hessian eigenvalues
            (lambda1, lambda2, lambda3) = eigenvalues

            # Compute sensitivity to deviation from a plate-like structure
            r_a = divide_nonzero(lambda2, lambda3) ** 2

            # Compute sensitivity to deviation from a blob-like structure
            r_b = divide_nonzero(lambda1,
                                 np.sqrt(np.abs(lambda2 * lambda3))) ** 2

            # Compute sensitivity to areas of high variance/texture/structure
            r_g = lambda1 ** 2 + lambda2 ** 2 + lambda3 ** 2

            # Compute output image for given (sigma) scale
            filtered = ((np.ones(shape) - np.exp(-r_a / alpha)) *
                        np.exp(-r_b / beta) *
                        (np.ones(shape) - np.exp(-r_g / gamma)))

            # Store results in (n+1)D matrices
            filtered_array[i] = filtered
            lambdas_array[i] = np.max([lambda2, lambda3], axis=0)

        else:

            # Check image dimensions
            assert_nD(image, [2, 3])

    # Remove background
    filtered_array[lambdas_array > 0] = 0

    # Return for every pixel the value of the (sigma) scale with the maximum
    # output pixel value
    return np.max(filtered_array, axis=0)


def hessian(image, scale_range=(1, 10), scale_step=2, beta1=None, beta2=None,
            alpha=0.5, beta=0.5, gamma=15, black_ridges=True):
    """Filter an image with the Hybrid Hessian filter.

    This filter can be used to detect continuous edges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the whole
    image containing such objects.

    Almost equal to Frangi filter, but uses alternative method of smoothing.
    Refer to _[1] to find the differences between Frangi and Hessian filters.

    Parameters
    ----------
    image : (N, M) ndarray
        Array with input image data.
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    alpha : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a plate-like structure.
    beta = beta1 : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a blob-like structure.
    gamma = beta2 : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to areas of high variance/texture/structure.
    black_ridges : boolean, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.

    Returns
    -------
    out : (N, M) ndarray
        Filtered image (maximum of pixels across all scales).

    Notes
    -----
    Written by Marc Schrijver (November 2001)
    Re-Written by D. J. Kroon University of Twente (May 2009) _[2]

    References
    ----------
    .. [1] Ng, C. C., Yap, M. H., Costen, N., & Li, B. (2014,). Automatic
        wrinkle detection using hybrid Hessian filter. In Asian Conference on
        Computer Vision (pp. 609-622). Springer International Publishing.
    .. [2] Kroon, D.J.: Hessian based Frangi vesselness filter.
    """

    # Check deprecated keyword parameters
    if beta1:
        warn("""Use keyword parameter 'beta' instead of 'beta1' which
                will be removed in version 0.16.""")
        beta = beta1

    if beta2:
        warn("""Use keyword parameter 'gamma' instead of 'beta2' which
                will be removed in version 0.16.""")
        gamma = beta2

    filtered = frangi(image, scale_range=scale_range, scale_step=scale_step,
                      beta1=None, beta2=None, alpha=alpha, beta=beta,
                      gamma=gamma, black_ridges=black_ridges)

    filtered[filtered <= 0] = 1
    return filtered
