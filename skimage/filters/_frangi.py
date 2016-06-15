import numpy as np

__all__ = ['frangi', 'hessian']


def _frangi_hessian_common_filter(image, scale, scale_step, beta1, beta2,
                               frangi_=True, black_ridges=True):
    """This is an intermediate function for Frangi and Hessian filters.

    Shares the common code for frangi and hessian functions.

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

    # Make matrices to store all filtered images
    filtered_list = []

    # Frangi filter for all sigmas
    for sigma in sigmas:
        # Make 2D hessian
        (Dxx, Dxy, Dyy) = hessian_matrix(image, sigma)

        # Correct for scale
        Dxx = (sigma ** 2) * Dxx
        Dxy = (sigma ** 2) * Dxy
        Dyy = (sigma ** 2) * Dyy

        # Calculate (abs sorted) eigenvalues and vectors
        (Lambda1, Lambda2) = hessian_matrix_eigvals(Dxx, Dxy, Dyy)

        # Compute some similarity measures
        Lambda1[Lambda1 == 0] = 1e-10
        Rb = (Lambda2 / Lambda1) ** 2
        S2 = Lambda1 ** 2 + Lambda2 ** 2

        # Compute the output image
        filtered = np.exp(-Rb / beta1) * (np.ones(np.shape(image)) -
                                          np.exp(-S2 / beta2))
        if frangi_:
            if black_ridges:
                filtered[Lambda1 < 0] = 0
            else:
                filtered[Lambda1 >= 0] = 0
        else:
            filtered[Lambda1 < 0] = 0

        # Store the results in 3D matrices
        filtered_list.append(filtered)
    return filtered_list


def frangi(image, scale=(1, 10), scale_step=2, beta1=0.5, beta2=15,
                  black_ridges=True):
    """Filter an image with the Frangi filter.

    This filter can be used to detect continous edges, e.g. vessels,
    wrinkles, rivers. It can be useful for calculation of fraction
    of image, containing such objects.

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

    filtered_list = _frangi_hessian_common_filter(image, scale, scale_step,
                                               beta1, beta2)


    # Return for every pixel the value of the scale(sigma) with the maximum
    # output pixel value

    return np.maximum.reduce(filtered_list)


def hessian(image, scale=(1, 10), scale_step=2, beta1=0.5, beta2=15):
    """Filter an image with the Hessian filter.

    This filter can be used to detect continous edges, e.g. vessels,
    wrinkles, rivers. It can be useful for calculation of fraction
    of image, containing such objects.

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

    frangi_ = False
    filtered_list = _frangi_hessian_common_filter(image, scale, scale_step,
                                                beta1, beta2, frangi_)

    # Return for every pixel the value of the scale(sigma) with the maximum
    # output pixel value
    out = np.maximum.reduce(filtered_list)
    out[out <= 0] = 1

    return out

