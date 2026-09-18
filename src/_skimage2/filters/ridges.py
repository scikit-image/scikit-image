"""
Ridge filters.

Ridge filters can be used to detect continuous edges, such as vessels,
neurites, wrinkles, rivers, and other tube-like structures. The present
class of ridge filters relies on the eigenvalues of the Hessian matrix of
image intensities to detect tube-like structures where the intensity changes
perpendicular but not along the structure.
"""

from warnings import warn

import numpy as np
from scipy import linalg

from _skimage2._shared.utils import _supported_float_type, check_nD


def meijering(
    image, sigmas=range(1, 10, 2), alpha=None, black_ridges=True, mode='reflect', cval=0
):
    """
    Filter an image with the Meijering neuriteness filter.

    This filter can be used to detect continuous ridges, e.g. neurites,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Calculates the eigenvalues of the Hessian to compute the similarity of
    an image region to neurites, according to the method described in [1]_.

    Parameters
    ----------
    image : (M, N[, ...]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter
    alpha : float, optional
        Shaping filter constant, that selects maximally flat elongated
        features.  The default, None, selects the optimal value -1/(ndim+1).
    black_ridges : bool, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    out : (M, N[, ...]) ndarray
        Filtered image (maximum of pixels across all scales).

    See also
    --------
    sato
    frangi
    hessian

    References
    ----------
    .. [1] Meijering, E., Jacob, M., Sarria, J. C., Steiner, P., Hirling, H.,
        Unser, M. (2004). Design and validation of a tool for neurite tracing
        and analysis in fluorescence microscopy images. Cytometry Part A,
        58(2), 167-176.
        :DOI:`10.1002/cyto.a.20022`
    """
    # Avoid circular import
    from ..feature.corner import hessian_matrix, hessian_matrix_eigvals

    image = image.astype(_supported_float_type(image.dtype), copy=False)
    if not black_ridges:  # Normalize to black ridges.
        image = -image

    if alpha is None:
        alpha = 1 / (image.ndim + 1)
    mtx = linalg.circulant([1, *[alpha] * (image.ndim - 1)]).astype(image.dtype)

    # Generate empty array for storing maximum value
    # from different (sigma) scales
    filtered_max = np.zeros_like(image)
    for sigma in sigmas:  # Filter for all sigmas.
        eigvals = hessian_matrix_eigvals(
            hessian_matrix(
                image, sigma, mode=mode, cval=cval, use_gaussian_derivatives=True
            )
        )
        # Compute normalized eigenvalues l_i = e_i + sum_{j!=i} alpha * e_j.
        vals = np.tensordot(mtx, eigvals, 1)
        # Get largest normalized eigenvalue (by magnitude) at each pixel.
        vals = np.take_along_axis(vals, abs(vals).argmax(0)[None], 0).squeeze(0)
        # Remove negative values.
        vals = np.maximum(vals, 0)
        # Normalize to max = 1 (unless everything is already zero).
        max_val = vals.max()
        if max_val > 0:
            vals /= max_val
        filtered_max = np.maximum(filtered_max, vals)

    return filtered_max  # Return pixel-wise max over all sigmas.


def sato(image, sigmas=range(1, 10, 2), black_ridges=True, mode='reflect', cval=0):
    """
    Filter an image with the Sato tubeness filter.

    This filter can be used to detect continuous ridges, e.g. tubes,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Defined only for 2-D and 3-D images. Calculates the eigenvalues of the
    Hessian to compute the similarity of an image region to tubes, according to
    the method described in [1]_.

    Parameters
    ----------
    image : (M, N[, P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter.
    black_ridges : bool, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    out : (M, N[, P]) ndarray
        Filtered image (maximum of pixels across all scales).

    See also
    --------
    meijering
    frangi
    hessian

    References
    ----------
    .. [1] Sato, Y., Nakajima, S., Shiraga, N., Atsumi, H., Yoshida, S.,
        Koller, T., ..., Kikinis, R. (1998). Three-dimensional multi-scale line
        filter for segmentation and visualization of curvilinear structures in
        medical images. Medical image analysis, 2(2), 143-168.
        :DOI:`10.1016/S1361-8415(98)80009-1`
    """
    # Avoid circular import
    from ..feature.corner import hessian_matrix, hessian_matrix_eigvals

    check_nD(image, [2, 3])  # Check image dimensions.
    image = image.astype(_supported_float_type(image.dtype), copy=False)
    if not black_ridges:  # Normalize to black ridges.
        image = -image

    # Generate empty array for storing maximum value
    # from different (sigma) scales
    filtered_max = np.zeros_like(image)
    for sigma in sigmas:  # Filter for all sigmas.
        eigvals = hessian_matrix_eigvals(
            hessian_matrix(
                image, sigma, mode=mode, cval=cval, use_gaussian_derivatives=True
            )
        )
        # Compute normalized tubeness (eqs. (9) and (22), ref. [1]_) as the
        # geometric mean of eigvals other than the lowest one
        # (hessian_matrix_eigvals returns eigvals in decreasing order), clipped
        # to 0, multiplied by sigma^2.
        eigvals = eigvals[:-1]
        vals = sigma**2 * np.prod(np.maximum(eigvals, 0), 0) ** (1 / len(eigvals))
        filtered_max = np.maximum(filtered_max, vals)
    return filtered_max  # Return pixel-wise max over all sigmas.


def _frangi_shape_norm(image, sigma, alpha, beta, mode, cval):
    """One scale of Frangi's vesselness, less the structuredness factor.

    Returns `shape`, the plate-and-blobness product of eqs. (13) and (15), and
    `norm`, the Frobenius norm `S` of eq. (12).  `shape` is zero where the
    eigenvalue signs rule the structure out.

    The two are returned separately rather than combined into the vesselness
    `V`, because the threshold `c` that `S` is measured against appears in
    eqs. (13) and (15) as a constant, and eq. (14) maximises over scale with
    it held fixed.  It can only be resolved once every scale is known.
    """
    # Avoid circular import
    from ..feature.corner import hessian_matrix, hessian_matrix_eigvals

    # For 2D image size I, J, eigvals is shape (2, I, J).
    eigvals = hessian_matrix_eigvals(
        hessian_matrix(
            image, sigma, mode=mode, cval=cval, use_gaussian_derivatives=True
        )
    )
    # Sort eigenvalues by magnitude.
    eigvals = np.take_along_axis(eigvals, np.argsort(np.abs(eigvals), axis=0), axis=0)
    # Normalised derivatives, eq. (2), at the gamma of unity the paper sets
    # "when no scale is preferred".  The Hessian is a second derivative, so it
    # carries sigma ** (2 * gamma) = sigma ** 2.  Without this the response of
    # an ideal ridge falls monotonically with sigma.
    eigvals = eigvals * sigma**2

    # Eqs. (13) (3D) and (15) (2D) are zero unless every eigenvalue but the
    # first has the sign the sought polarity implies.  `frangi` negates the
    # image for `black_ridges=False`, so the sign wanted here is always
    # positive.
    wanted = np.all(eigvals[1:] > 0, axis=0)  # N-D mask.
    selected_eigvals = eigvals[:, wanted]
    lambda1, lambda2 = selected_eigvals[0:2]
    shape = np.zeros_like(image, order='C')
    if image.ndim == 2:
        # No plate factor in 2-D; implied by eq. (15).
        r_b_sq = (lambda1 / lambda2) ** 2  # eq. (15)
        shape[wanted] = np.exp(-r_b_sq / (2 * beta**2))  # blobness
    else:  # ndim == 3
        lambda3 = selected_eigvals[2]
        r_a_sq = (lambda2 / lambda3) ** 2  # eq. (11)
        r_b_sq = lambda1**2 / (lambda2 * lambda3)  # eq. (10)
        plateness = 1.0 - np.exp(-r_a_sq / (2 * alpha**2))  # eq. (13)
        blobness = np.exp(-r_b_sq / (2 * beta**2))
        shape[wanted] = plateness * blobness
    return shape, np.sqrt(np.sum(eigvals**2, axis=0))


def frangi(
    image,
    sigmas=range(1, 10, 2),
    scale_range=None,
    scale_step=None,
    alpha=0.5,
    beta=0.5,
    gamma=None,
    black_ridges=True,
    mode='reflect',
    cval=0,
):
    """
    Filter an image with the Frangi vesselness filter.

    This filter can be used to detect continuous ridges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the
    whole image containing such objects.

    Defined only for 2-D and 3-D images. Calculates the eigenvalues of the
    Hessian to compute the similarity of an image region to vessels, according
    to the method described in [1]_.

    Parameters
    ----------
    image : (M, N[, P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter, i.e.,
        np.arange(scale_range[0], scale_range[1], scale_step)
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    alpha : float, optional
        Frangi correction constant that adjusts the filter's sensitivity to
        deviation from a plate-like structure.  It has no effect on 2-D images:
        the plate-sensitivity factor of eq. (13) is absent from the 2-D
        vesselness of eq. (15) in [1]_.
    beta : float, optional
        Frangi correction constant that adjusts the filter's sensitivity to
        deviation from a blob-like structure.
    gamma : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to areas of high variance/texture/structure.  This is the
        filter's only absolute contrast reference; every other quantity is a
        ratio.

        The default, None, resolves it to half the largest Hessian norm over
        the whole image and every scale, which is the heuristic [1]_
        recommends.  Being a whole-image statistic, it makes the result at any
        pixel depend on every other pixel: one bright speck anywhere changes
        the output everywhere.  Pass an explicit `gamma` where that matters.
        [1]_ anticipates this, expecting the threshold "can be fixed for a
        given application where images are routinely acquired according to a
        standard protocol".

        .. versionchanged:: 0.20
            The default, None, uses half of the maximum Hessian norm.

    black_ridges : bool, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    out : (M, N[, P]) ndarray
        Filtered image (maximum of pixels across all scales).

    .. versionchanged:: 0.20
        The implementation got rewritten and gives different output values wrt
        the previous implementation (backwards incompatible change).
        The filter is now set to zero whenever one of the Hessian eigenvalues
        has a sign which is incompatible with a ridge of the desired polarity.

    .. versionchanged:: 0.27
        The Hessian is again normalised by ``sigma ** 2``, as it was before
        version 0.20 and as ``sato`` has been throughout.  Without it the
        response of a ridge falls with ``sigma``, so the smallest scale always
        won and wide structures scored far below narrow ones.  Output values
        change on every image.

    .. versionchanged:: 0.27
        ``gamma=None`` is resolved from every scale rather than from
        ``sigmas[0]``, so the result no longer depends on the order of
        ``sigmas``.  For an ascending ``sigmas`` the output is unchanged.

    Notes
    -----
    The derivatives are normalised across scale as in eq. (2) of [1]_, with
    Lindeberg's ``gamma`` set to unity as that paper prescribes "when no scale
    is preferred".  The Hessian is a second derivative, so it carries a factor
    ``sigma ** 2``.  This is what lets the maximum over scales in eq. (14)
    select a structure's width; ``sato`` uses the same convention.

    Earlier versions of this filter were implemented by Marc Schrijver,
    (November 2001), D. J. Kroon, University of Twente (May 2009) [2]_, and
    D. G. Ellis (January 2017) [3]_.

    See also
    --------
    meijering
    sato
    hessian

    References
    ----------
    .. [1] Frangi, A. F., Niessen, W. J., Vincken, K. L., & Viergever, M. A.
        (1998,). Multiscale vessel enhancement filtering. In International
        Conference on Medical Image Computing and Computer-Assisted
        Intervention (pp. 130-137). Springer Berlin Heidelberg.
        :DOI:`10.1007/BFb0056195`
    .. [2] Kroon, D. J.: Hessian based Frangi vesselness filter.
    .. [3] Ellis, D. G.: https://github.com/ellisdg/frangi3d/tree/master/frangi
    """
    if scale_range is not None and scale_step is not None:
        warn(
            'Use keyword parameter `sigmas` instead of `scale_range` and '
            '`scale_range` which will be removed in version 0.17.',
            stacklevel=2,
        )
        sigmas = np.arange(scale_range[0], scale_range[1], scale_step)

    check_nD(image, [2, 3])  # Check image dimensions.
    image = image.astype(_supported_float_type(image.dtype), copy=False)
    if not black_ridges:  # Normalize to black ridges.
        image = -image

    # `gamma` is a constant of eqs. (13) and (15), and eq. (14) maximises over
    # scale with it held fixed, so it cannot be resolved from one scale inside
    # the loop. Resolving it needs only each scale's *maximum* of `S`, which is
    # a scalar.
    if gamma is None:
        peaks = [
            _frangi_shape_norm(image, sigma, alpha, beta, mode, cval)[1].max()
            for sigma in sigmas
        ]
        # Half the largest Hessian norm, over every scale given.
        gamma = max(peaks) / 2 if peaks else 1.0
        if gamma == 0:
            gamma = 1  # If S == 0 everywhere, gamma doesn't matter.

    # Filtered image, eqs. (13) and (15), then the maximum over scales.  With
    # `gamma` known, each scale is fused and discarded as it is computed.
    filtered_max = np.zeros_like(image)
    for sigma in sigmas:
        shape, norm = _frangi_shape_norm(image, sigma, alpha, beta, mode, cval)
        structuredness = 1.0 - np.exp(
            -(norm**2) / (2 * gamma**2), dtype=image.dtype
        )
        filtered_max = np.maximum(filtered_max, shape * structuredness)
    return filtered_max  # Return pixel-wise max over all sigmas.


def hessian(
    image,
    sigmas=range(1, 10, 2),
    scale_range=None,
    scale_step=None,
    alpha=0.5,
    beta=0.5,
    gamma=15,
    black_ridges=True,
    mode='reflect',
    cval=0,
):
    """Filter an image with the Hybrid Hessian filter.

    This filter can be used to detect continuous edges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the whole
    image containing such objects.

    Defined only for 2-D and 3-D images. Almost equal to Frangi filter, but
    uses alternative method of smoothing. Refer to [1]_ to find the differences
    between Frangi and Hessian filters.

    Parameters
    ----------
    image : (M, N[, P]) ndarray
        Array with input image data.
    sigmas : iterable of floats, optional
        Sigmas used as scales of filter, i.e.,
        np.arange(scale_range[0], scale_range[1], scale_step)
    scale_range : 2-tuple of floats, optional
        The range of sigmas used.
    scale_step : float, optional
        Step size between sigmas.
    beta : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to deviation from a blob-like structure.
    gamma : float, optional
        Frangi correction constant that adjusts the filter's
        sensitivity to areas of high variance/texture/structure.
    black_ridges : bool, optional
        When True (the default), the filter detects black ridges; when
        False, it detects white ridges.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.

    Returns
    -------
    out : (M, N[, P]) ndarray
        Filtered image (maximum of pixels across all scales).

    Notes
    -----
    Written by Marc Schrijver (November 2001)
    Re-Written by D. J. Kroon University of Twente (May 2009) [2]_

    See also
    --------
    meijering
    sato
    frangi

    References
    ----------
    .. [1] Ng, C. C., Yap, M. H., Costen, N., & Li, B. (2014,). Automatic
        wrinkle detection using hybrid Hessian filter. In Asian Conference on
        Computer Vision (pp. 609-622). Springer International Publishing.
        :DOI:`10.1007/978-3-319-16811-1_40`
    .. [2] Kroon, D. J.: Hessian based Frangi vesselness filter.
    """
    filtered = frangi(
        image,
        sigmas=sigmas,
        scale_range=scale_range,
        scale_step=scale_step,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges,
        mode=mode,
        cval=cval,
    )

    filtered[filtered <= 0] = 1
    return filtered
