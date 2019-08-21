from warnings import warn

from ..exposure import match_histograms as mh

def match_histograms(image, reference, multichannel=False):
    warn('DEPRECATED: skimage.transform.match_histograms has been moved to '
         'skimage.exposure.match_histograms. It will be removed from '
         'skimage.transform in version 0.18.', stacklevel=2)
    return mh(image, reference, multichannel=False)


if mh.__doc__ is not None:
    match_histograms.__doc__ = mh.__doc__ + """
    Warns
    -----
    Deprecated:
        .. versionadded:: 0.16

        This function is deprecated and will be removed in scikit-image 0.18.
        Please use ``match_histograms`` from ``exposure`` module instead.

    See also
    --------
    skimage.exposure.match_histograms
    """
