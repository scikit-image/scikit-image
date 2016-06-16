import numpy as np

__all__ = ['frangi', 'hessian']


def _frangi_hessian_common_filter(image, scale, scale_step, beta1, beta2,
                                  frangi_=True, black_ridges=True):
    """This is an intermediate function for Frangi and Hessian filters.

    Shares the common code for Frangi and Hessian functions.

    Parameters
    ----------
    image : (N, M) ndarray
        Array with input image data.
    scale : tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    beta1 : float, optional
        Frangi correction constant.
    beta2 : float, optional
        Frangi correction constant.
    black_ridges : boolean, optional
        If True (default), detects black ridges, if False - white ones.

    Returns
    -------
    filtered_list : list
        List of pre-filtered images.

    """

    # Import has to be here due to circular import error
    from ..feature import hessian_matrix, hessian_matrix_eigvals

    sigmas = np.arange(scale[0], scale[1], scale_step)

    if np.any(np.asarray(sigmas) < 0.0):
        raise ValueError("Sigma values less than zero are not valid")

    beta1 = 2 * beta1 ** 2
    beta2 = 2 * beta2 ** 2

    filtered_array = np.zeros(len(sigmas), np.shape(image)[0], np.shape(image)[1])
    lambdas_array = np.zeros(len(sigmas))

    # Filtering for all sigmas
    for i in range(len(sigmas)):
        sigma = sigmas[i]
        # Make 2D hessian
        (Dxx, Dxy, Dyy) = hessian_matrix(image, sigma)

        # Correct for scale
        Dxx = (sigma ** 2) * Dxx
        Dxy = (sigma ** 2) * Dxy
        Dyy = (sigma ** 2) * Dyy

        # Calculate (abs sorted) eigenvalues and vectors
        (lambda1, lambda2) = hessian_matrix_eigvals(Dxx, Dxy, Dyy)

        # Compute some similarity measures
        lambda1[lambda1 == 0] = 1e-10
        rb = (lambda2 / lambda1) ** 2
        s2 = lambda1 ** 2 + lambda2 ** 2

        # Compute the output image
        filtered = np.exp(-rb / beta1) * (np.ones(np.shape(image)) -
                                          np.exp(-s2 / beta2))

        # Store the results in 3D matrices
        filtered_array[i] = filtered
        lambdas_array[i] = lambda1
    return filtered_array, lambdas_array


def frangi(image, scale=(1, 10), scale_step=2, beta1=0.5, beta2=15,
           black_ridges=True):
    """Filter an image with the Frangi filter.

    This filter can be used to detect continous edges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Calculates the eigenvectors of the Hessian to compute the likeliness of
    an image region to vessels, according to the method described in _[1].

    Parameters
    ----------
    image : (N, M) ndarray
        Array with input image data.
    scale : tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    beta1 : float, optional
        Frangi correction constant.
    beta2 : float, optional
        Frangi correction constant.
    black_ridges : boolean, optional
        Detect black ridges (default) set to true, for
        white ridges set to false.

    Returns
    -------
    out : (N, M) ndarray
        Filtered image (maximum of pixels across all scales).

    Notes
    -----
    Written by Marc Schrijver, 2/11/2001
    Re-Written by D. J. Kroon University of Twente (May 2009)

    References
    ----------
    .. [1] A. Frangi, W. Niessen, K. Vincken, and M. Viergever. "Multiscale
    vessel enhancement filtering," In LNCS, vol. 1496, pages 130-137,
    Germany, 1998. Springer-Verlag.
    .. [2] Kroon, D.J.: Hessian based frangi vesselness filter.
    .. [3] http://mplab.ucsd.edu/tutorials/gabor.pdf.
    """

    (filtered_array, lambdas_array) = _frangi_hessian_common_filter(
                                      image, scale, scale_step, beta1, beta2)

    for i in range(len(filtered_array)):
        filtered = filtered_array[i]
        lambda1 = lambdas_array[i]
        if black_ridges:
            filtered[lambda1 < 0] = 0
        else:
            filtered[lambda1 >= 0] = 0
        filtered_array[i][0] = filtered

    # Return for every pixel the value of the scale(sigma) with the maximum
    # output pixel value

    return np.max(filtered_array, axis=0)[0]


def hessian(image, scale=(1, 10), scale_step=2, beta1=0.5, beta2=15):
    """Filter an image with the Hessian filter.

    This filter can be used to detect continous edges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the whole
    image containing such objects

    Almost equal to frangi filter, but uses alternative method of smoothing.
    Address _[1] to find the differences between Frangi and Hessian filters.

    Parameters
    ----------
    image : (N, M) ndarray
        Array with input image data.
    scale : tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    beta1 : float, optional
        Frangi correction constant.
    beta2 : float, optional
        Frangi correction constant.

    Returns
    -------
    out : (N, M) ndarray
        Filtered image (maximum of pixels across all scales).

    Notes
    -----
    Written by Marc Schrijver, 2/11/2001
    Re-Written by D. J. Kroon University of Twente (May 2009)

    References
    ----------
    .. [1] Choon-Ching Ng, Moi Hoon Yap, Nicholas Costen and Baihua Li,
    "Automatic Wrinkle Detection using Hybrid Hessian Filter".
    """

    (filtered_array, lambdas_array) = _frangi_hessian_common_filter(
                                      image, scale, scale_step, beta1, beta2)

    for i in range(len(filtered_array)):
        filtered = filtered_array[i]
        lambda1 = lambdas_array[i]
        filtered[lambda1 < 0] = 0
        filtered_array[i][0] = filtered

    # Return for every pixel the value of the scale(sigma) with the maximum
    # output pixel value
    out = np.max(filtered_array, axis=0)
    out[out <= 0] = 1

    return out[0]

