from _skimage2.feature.corner import (
    corner_fast as corner_fast,
    corner_foerstner as corner_foerstner,
    corner_harris as corner_harris,
    corner_kitchen_rosenfeld as corner_kitchen_rosenfeld,
    corner_moravec as corner_moravec,
    corner_orientations as corner_orientations,
    corner_shi_tomasi as corner_shi_tomasi,
    corner_subpix as corner_subpix,
    hessian_matrix_det as hessian_matrix_det,
    hessian_matrix_eigvals as hessian_matrix_eigvals,
    shape_index as shape_index,
    structure_tensor as structure_tensor,
    structure_tensor_eigenvalues as structure_tensor_eigenvalues,
)  # noqa: F401

__all__ = [
    'corner_fast',
    'corner_foerstner',
    'corner_harris',
    'corner_kitchen_rosenfeld',
    'corner_moravec',
    'corner_orientations',
    'corner_peaks',
    'corner_shi_tomasi',
    'corner_subpix',
    'hessian_matrix',
    'hessian_matrix_det',
    'hessian_matrix_eigvals',
    'shape_index',
    'structure_tensor',
    'structure_tensor_eigenvalues',
]

import numpy as np

from _skimage2.feature.corner import corner_peaks as ski2_corner_peaks
from _skimage2.feature.corner import hessian_matrix as ski2_hessian_matrix

from skimage._migration import ski2_migration_decorator


@ski2_migration_decorator(
    """\
``%(qname_old)s`` will be removed in scikit-image 2. Please use
``skimage2.feature.peak_local_max`` instead.
""",
    qname_old='skimage.feature.corner_peaks',
)
def corner_peaks(
    image,
    min_distance=1,
    threshold_abs=None,
    threshold_rel=None,
    exclude_border=True,
    indices=True,
    num_peaks=np.inf,
    footprint=None,
    labels=None,
    *,
    num_peaks_per_label=np.inf,
    p_norm=np.inf,
):
    """Find peaks in corner measure response image.

    This differs from `skimage.feature.peak_local_max` in that it suppresses
    multiple connected peaks with the same accumulator value.

    Parameters
    ----------
    image : ndarray of shape (M, N)
        Input image.
    min_distance : int, optional
        The minimal allowed distance separating peaks.
    * : *
        See :py:meth:`skimage.feature.peak_local_max`.
    p_norm : float
        Which Minkowski p-norm to use. Should be in the range [1, inf].
        A finite large p may cause a ValueError if overflow can occur.
        ``inf`` corresponds to the Chebyshev distance and 2 to the
        Euclidean distance.

    Returns
    -------
    output : ndarray or ndarray of bools

        * If `indices = True`  : (row, column, ...) coordinates of peaks.
        * If `indices = False` : Boolean array shaped like `image`, with peaks
          represented by True values.

    See also
    --------
    skimage.feature.peak_local_max

    Notes
    -----
    .. versionchanged:: 0.18
        The default value of `threshold_rel` has changed to None, which
        corresponds to letting `skimage.feature.peak_local_max` decide on the
        default. This is equivalent to `threshold_rel=0`.

    The `num_peaks` limit is applied before suppression of connected peaks.
    To limit the number of peaks after suppression, set `num_peaks=np.inf` and
    post-process the output of this function.

    Examples
    --------
    >>> from skimage.feature import peak_local_max
    >>> response = np.zeros((5, 5))
    >>> response[2:4, 2:4] = 1
    >>> response
    array([[0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0.],
           [0., 0., 1., 1., 0.],
           [0., 0., 1., 1., 0.],
           [0., 0., 0., 0., 0.]])
    >>> peak_local_max(response)
    array([[2, 2],
           [2, 3],
           [3, 2],
           [3, 3]])
    >>> corner_peaks(response)
    array([[2, 2]])

    """
    # Allow exclude_border=True|False
    exclude_border = (
        0
        if exclude_border is False
        else (int(np.floor(min_distance)) if exclude_border is True else exclude_border)
    )
    return ski2_corner_peaks(
        image,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
        indices=indices,
        num_peaks=num_peaks,
        footprint=footprint,
        labels=labels,
        num_peaks_per_label=num_peaks_per_label,
        p_norm=p_norm,
    )


@ski2_migration_decorator(
    """\
    ``%(qname_old)s`` is deprecated in favor of ``%(qname_new)s`` with new
    behavior:

    * The default method for calculating the Hessian has changed from finite
      differences on the Gaussian-smoothed image
      (``use_gaussian_derivatives=False``) to the more accurate convolution
      with Gaussian derivatives (``use_gaussian_derivatives=True``).
    * Parameter `use_gaussian_derivatives` is now keyword-only.

    To keep the old (``skimage``, v1.x) default behavior after switching to
    ``skimage2``, pass ``use_gaussian_derivatives=False`` explicitly::

        skimage.feature.hessian_matrix(image, ...)

    Becomes ::

        skimage2.feature.hessian_matrix(image, ..., use_gaussian_derivatives=False)

    Other parameters can be left unchanged, but pass
    `use_gaussian_derivatives` by keyword.
    """,
    qname_old='skimage.feature.hessian_matrix',
)
def hessian_matrix(
    image, sigma=1, mode='constant', cval=0, order='rc', use_gaussian_derivatives=None
):
    r"""Compute the Hessian matrix.

    In 2D, the Hessian matrix is defined as::

        H = [Hrr Hrc]
            [Hrc Hcc]

    which is computed by convolving the image with the second derivatives
    of the Gaussian kernel in the respective r- and c-directions, or by
    finite differences on the Gaussian-smoothed image (the default).

    The implementation here also supports n-dimensional data.

    Parameters
    ----------
    image : ndarray
        Input image.
    sigma : float
        Standard deviation used for the Gaussian kernel, which is used as
        weighting function for the auto-correlation matrix.
    mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
        How to handle values outside the image borders.
    cval : float, optional
        Used in conjunction with mode 'constant', the value outside
        the image boundaries.
    order : {'rc', 'xy'}, optional
        For 2D images, this parameter allows for the use of reverse or forward
        order of the image axes in gradient computation. 'rc' indicates the use
        of the first axis initially (Hrr, Hrc, Hcc), whilst 'xy' indicates the
        usage of the last axis initially (Hxx, Hxy, Hyy). Images with higher
        dimension must always use 'rc' order.
    use_gaussian_derivatives : bool or None, optional
        If True, compute the Hessian by convolving the image with derivatives
        of the Gaussian kernel.  If False, compute the Hessian by finite
        differences on the Gaussian-smoothed image.  None (the default) is
        the same as False.  See Notes.

    Returns
    -------
    H_elems : list of ndarray
        Upper-diagonal elements of the hessian matrix for each pixel in the
        input image. In 2D, this will be a three element list containing [Hrr,
        Hrc, Hcc]. In nD, the list will contain ``(n**2 + n) / 2`` arrays.

    Notes
    -----
    The distributive property of derivatives and convolutions allows us to
    restate the derivative of an image, I, smoothed with a Gaussian kernel, G,
    as the convolution of the image with the derivative of G.

    .. math::

        \frac{\partial }{\partial x_i}(I * G) =
        I * \left( \frac{\partial }{\partial x_i} G \right)

    When ``use_gaussian_derivatives`` is ``True``, this property is used to
    compute the second order derivatives that make up the Hessian matrix.

    When ``use_gaussian_derivatives`` is ``False`` or None (the default),
    simple finite differences on a Gaussian-smoothed image are used instead.
    This is less accurate than the Gaussian derivative method.

    Examples
    --------
    >>> from skimage.feature import hessian_matrix
    >>> square = np.zeros((5, 5))
    >>> square[2, 2] = 4
    >>> Hrr, Hrc, Hcc = hessian_matrix(square, sigma=0.1, order='rc',
    ...                                use_gaussian_derivatives=False)
    >>> Hrc
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0., -1.,  0.,  1.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])

    """
    if use_gaussian_derivatives is None:
        use_gaussian_derivatives = False
    return ski2_hessian_matrix(
        image,
        sigma=sigma,
        mode=mode,
        cval=cval,
        order=order,
        use_gaussian_derivatives=use_gaussian_derivatives,
    )


from skimage._doctest_adapters import adapt_doctests  # noqa: E402

adapt_doctests(globals())
