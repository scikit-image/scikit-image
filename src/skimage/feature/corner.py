import numpy as np
from scipy import spatial

from _skimage2.feature.corner import (
    corner_fast as corner_fast,
    corner_foerstner as corner_foerstner,
    corner_harris as corner_harris,
    corner_kitchen_rosenfeld as corner_kitchen_rosenfeld,
    corner_moravec as corner_moravec,
    corner_orientations as corner_orientations,
    corner_shi_tomasi as corner_shi_tomasi,
    corner_subpix as corner_subpix,
    hessian_matrix as hessian_matrix,
    hessian_matrix_det as hessian_matrix_det,
    hessian_matrix_eigvals as hessian_matrix_eigvals,
    shape_index as shape_index,
    structure_tensor as structure_tensor,
    structure_tensor_eigenvalues as structure_tensor_eigenvalues,
)  # noqa: F401

from .peak import peak_local_max


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
    if np.isinf(num_peaks):
        num_peaks = None
    if np.isinf(num_peaks_per_label):
        num_peaks_per_label = None

    # Get the coordinates of the detected peaks
    coords = peak_local_max(
        image,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        threshold_rel=threshold_rel,
        exclude_border=exclude_border,
        num_peaks=None,  # Limiting to `num_peaks` is done in this function
        footprint=footprint,
        labels=labels,
        num_peaks_per_label=num_peaks_per_label,
        p_norm=p_norm,
    )

    if len(coords):
        # Use KDtree to find the peaks that are too close to each other
        tree = spatial.cKDTree(coords)

        rejected_peaks_indices = set()
        for idx, point in enumerate(coords):
            if idx not in rejected_peaks_indices:
                candidates = tree.query_ball_point(point, r=min_distance, p=p_norm)
                candidates.remove(idx)
                rejected_peaks_indices.update(candidates)

        # Remove the peaks that are too close to each other
        coords = np.delete(coords, tuple(rejected_peaks_indices), axis=0)

        if num_peaks is not None and len(coords) > num_peaks:
            # Sort by intensity (highest first) before applying the `num_peaks` limit.
            # Without labels, `peak_local_max` already returns peaks in intensity order,
            # but with labels the peaks are grouped per label, so taking the first
            # `num_peaks` would bias toward the lowest label IDs.
            intensities = image[tuple(coords.T)]
            order = np.argsort(-intensities, stable=True)
            order = order[:num_peaks]
            coords = coords[order, :]

    if indices:
        return coords

    peaks = np.zeros_like(image, dtype=bool)
    peaks[tuple(coords.T)] = True

    return peaks
