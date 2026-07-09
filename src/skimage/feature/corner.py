import numpy as np
from scipy import spatial

from _skimage2.feature._corner import (
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

from .._migration import ski2_migration_decorator
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


@ski2_migration_decorator(
    r"""
    ``%(qname_old)s`` is deprecated in favor of
    ``%(qname_new)s`` with new behavior:

    * Peaks are removed when `< min_distance`, was `<= min_distance`
    * Parameter `indices` is removed
    * Parameter `p_norm` defaults to 2 (Euclidean distance),
      was `numpy.inf` (Chebyshev distance)
    * Parameter `exclude_border` defaults to 1, was ``True``
    * Parameter `exclude_border` no longer accepts `False` and `True`,
      pass 0 instead of `False`, or the value of `min_distance` instead of `True`
    * Parameters after `image` are keyword-only

    To keep the old behavior when switching to `skimage2`, update your call
    according to the following cases:

    <!--- cond-start: warning -->
    * `min_distance` not passed, use `min_distance=np.nextafter(1, np.inf)
    * `min_distance=<number>`, use `min_distance=np.nextafter(<number>, np.inf)
    * `exclude_border` not passed, use `exclude_border=<value_of_min_distance>`
    * `exclude_border=True`, same as above
    * `exclude_border=False`, use `exclude_border=0`
    * `exclude_border=<int>`, no change necessary
    * `p_norm` not passed, use `p_norm=numpy.inf`
    * `p_norm=<float>`, no change necessary
    * `indices=True` or not passed, no change necessary
    * `indices=False`, boolean mask with:
          coords = ski2.feature.peak_local_max(...)
          peaks = np.zeros_like(image, dtype=bool)
          peaks[tuple(coords.T)] = True
    <!--- cond-end -->
    <!--- cond-start: doc -->
    .. list-table::
        :header-rows: 1

        - - In `skimage`
          - In `skimage2`

        - - `min_distance` not passed (default)
          - Use ``min_distance=numpy.nextafter(1, numpy.inf)``

        - - ``min_distance=<number>``
          - Use ``min_distance=numpy.nextafter(<number>, numpy.inf)``

        - - `exclude_border` not passed (default)
          - Assign it the same value as `min_distance` which may be its default
            value ``1``. If `min_distance` is a float,
            use ``int(np.floor(min_distance))``

        - - ``exclude_border=True``
          - Same as above in the default case.

        - - ``exclude_border=False``
          - Use ``min_distance=0``.

        - - ``exclude_border=<int>``
          - No change necessary.

        - - ``p_norm`` not passed (default)
          - Pass the Skimage 1 default explicitly with ``p_norm=numpy.inf``.

        - - ``p_norm=<float>``
          - No change necessary.

        - - ``indices=True`` or not passed (default)
          - No change necessary.

        - - ``indices=False``
          - Reconstruct peak mask with::

               coords = ski2.feature.peak_local_max(...)
               peaks = np.zeros_like(image, dtype=bool)
               peaks[tuple(coords.T)] = True

    Other keyword parameters can be left unchanged.

    >>> import numpy as np
    >>> import skimage as ski1
    >>> import skimage2 as ski2

    >>> image = ski2.data.camera()

    >>> res1 = ski1.feature.corner_peaks(image)
    >>> res2 = ski2.feature.peak_local_max(
    ...     image,
    ...     min_distance=np.nextafter(1, np.inf),
    ...     exclude_border=1,
    ...     p_norm=np.inf,
    ... )
    >>> np.testing.assert_equal(res1, res2)

    >>> res1 = ski1.feature.corner_peaks(image, min_distance=10, indices=False)
    >>> coords2 = ski2.feature.peak_local_max(
    ...     image,
    ...     min_distance=np.nextafter(10, np.inf),
    ...     exclude_border=10,
    ...     p_norm=np.inf
    ... )
    >>> res2 = np.zeros_like(image, dtype=bool)
    >>> res2[tuple(coords2.T)] = True
    >>> np.testing.assert_equal(res1, res2)

    <!--- cond-end -->
    """,
    qname_old='skimage.feature.corner_peaks',
    qname_new='skimage2.feature.peak_local_max',
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
