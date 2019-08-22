from warnings import warn
from ..metrics._structural_similarity import structural_similarity

__all__ = ['compare_ssim']


def compare_ssim(X, Y, win_size=None, gradient=False,
                 data_range=None, multichannel=False, gaussian_weights=False,
                 full=False, **kwargs):
    warn('DEPRECATED: skimage.measure.compare_ssim has been moved to '
         'skimage.metrics.structural_similarity. It will be removed from '
         'skimage.measure in version 0.18.', stacklevel=2)
    return structural_similarity(X, Y, win_size, gradient,
                                 data_range, multichannel,
                                 gaussian_weights, full, **kwargs)


if structural_similarity.__doc__ is not None:
    compare_ssim.__doc__ = structural_similarity.__doc__ + """

    Warns
    -----
    Deprecated:
        .. versionadded:: 0.16

        This function is deprecated and will be
        removed in scikit-image 0.18. Please use the function named
        ``structural_similarity`` from the ``metrics`` module instead.


    See also
    --------
    skimage.metrics.structural_similarity
    """
