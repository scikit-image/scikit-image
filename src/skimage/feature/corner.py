import numpy as np

from _skimage2.feature.corner import *  # noqa: F403
from _skimage2.feature.corner import corner_peaks as ski2_corner_peaks
from _skimage2.feature.corner import __doc__  # noqa: F401

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
    >>> from _skimage2.feature import peak_local_max
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
    exclude_border = 0 if exclude_border is False else (
        int(np.floor(min_distance)) if exclude_border is True
        else exclude_border)
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
