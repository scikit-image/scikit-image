"""Ridge filters

Ridge filters can be used to detect continuous edges, such as vessels,
neurites, wrinkles, rivers, and other tube-like structures. The present
class of ridge filters relies on the eigenvalues of the Hessian matrix of
image intensities to detect tube-like structures where the intensity changes
perpendicular but not along the structure.

"""

import numpy as np

import _skimage2 as ski2
from _skimage2.filters.ridges import (
    meijering as meijering,
    sato as sato,
)  # noqa: F401

from .._migration import ski2_migration_decorator

__all__ = [
    'frangi',
    'hessian',
    'meijering',
    'sato',
]


def _resolve_sigmas(sigmas, scale_range, scale_step):
    """Prefer deprecated scale_range/scale_step when both are given."""
    if scale_range is not None and scale_step is not None:
        return np.arange(scale_range[0], scale_range[1], scale_step)
    return sigmas


_SCALE_MIGRATION = """\
``%(qname_old)s`` is deprecated in favor of
``%(qname_new)s`` with new behavior:

* We removed parameters `scale_range` and `scale_step`.
* Pass the scale list with `sigmas` instead.

To keep the old (``skimage``, v1.x) behavior of `scale_range` / `scale_step`
after switching to ``skimage2``::

    %(qname_old)s(image, scale_range=(1, 10), scale_step=2)

Becomes ::

    %(qname_new)s(image, sigmas=np.arange(1, 10, 2))

Other keyword parameters can be left unchanged.
"""


@ski2_migration_decorator(
    _SCALE_MIGRATION,
    qname_old="skimage.filters.frangi",
)
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
    r"""
    Filter an image with the Frangi vesselness filter.

    This filter can be used to detect continuous ridges, e.g. vessels,
    wrinkles, rivers. It can be used to calculate the fraction of the whole
    image containing such objects.

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
        The range of sigmas used.  Ignored unless `scale_step` is also given.
        Use `sigmas` instead; removed in ``skimage2``.
    scale_step : float, optional
        Step size between sigmas.  Ignored unless `scale_range` is also given.
        Use `sigmas` instead; removed in ``skimage2``.
    alpha : float, optional
        Frangi correction constant that adjusts the filter's sensitivity to
        deviation from a plate-like structure.  It has no effect on 2-D images:
        the plate-sensitivity factor of eq. (13) is absent from the 2-D
        vesselness of eq. (15) in [1]_.
    beta : float, optional
        Frangi correction constant that adjusts the filter's sensitivity to
        deviation from a blob-like structure.
    gamma : float, optional
        Frangi correction constant that adjusts the filter's sensitivity to
        areas of high variance/texture/structure.  This is the threshold
        :math:`c` of eqs. (13) and (15) in [1]_, not the :math:`\gamma` of
        eq. (2), which is the scale-normalisation exponent (see Notes).  It is
        the filter's only absolute contrast reference; every other quantity is
        a ratio.

        The default, None, resolves it to half the largest Hessian norm, as
        [1]_ suggests, taken over the whole image and every scale.  Being a
        whole-image statistic, it makes the result at any pixel depend on every
        other pixel: one bright speck anywhere changes the output everywhere.
        Pass an explicit `gamma` where that matters.

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

    .. versionchanged:: 0.28
        The Hessian is again normalised by ``sigma ** 2``, as it was before
        version 0.20 and as ``sato`` has been throughout.  Without it the
        response of a ridge is lower with higher ``sigma``, so the smallest scale always
        won and wide structures scored far below narrow ones.  Output values
        change on every image.

    .. versionchanged:: 0.28
        ``gamma=None`` is resolved from every scale rather than from
        ``sigmas[0]``, so the result no longer depends on the order of
        ``sigmas``.

    Notes
    -----
    The per-sigma derivatives used in [1]_ are Lindeberg's
    :math:`\gamma`-normalised derivatives, as in eq. (2), with :math:`\sigma`
    the paper's scale :math:`s`.  The Hessian is a second derivative, so it
    is scaled by :math:`\sigma^{2\gamma}`.  [1]_ sets :math:`\gamma` to 1 when
    "no scale is preferred", so the factor is :math:`\sigma^2`.  This Lindeberg
    :math:`\gamma` is unrelated to the `gamma` parameter.  The normalisation is
    what lets the maximum over scales in eq. (14) of [1]_ select a structure's
    width; ``sato`` uses the same convention.

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
    return ski2.filters.frangi(
        image,
        sigmas=_resolve_sigmas(sigmas, scale_range, scale_step),
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges,
        mode=mode,
        cval=cval,
    )


@ski2_migration_decorator(
    _SCALE_MIGRATION,
    qname_old="skimage.filters.hessian",
)
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
        The range of sigmas used.  Ignored unless `scale_step` is also given.
        Use `sigmas` instead; removed in ``skimage2``.
    scale_step : float, optional
        Step size between sigmas.  Ignored unless `scale_range` is also given.
        Use `sigmas` instead; removed in ``skimage2``.
    alpha : float, optional
        Frangi correction constant that adjusts the filter's sensitivity to
        deviation from a plate-like structure.
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
    return ski2.filters.hessian(
        image,
        sigmas=_resolve_sigmas(sigmas, scale_range, scale_step),
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges,
        mode=mode,
        cval=cval,
    )


from skimage._doctest_adapters import adapt_doctests  # noqa: E402

adapt_doctests(globals())
