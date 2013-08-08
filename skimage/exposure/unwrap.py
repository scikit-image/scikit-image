import numpy as np
import warnings

from ._unwrap_naive import unwrap_naive_1d
from ._unwrap_2d import unwrap_2d
from ._unwrap_3d import unwrap_3d
from ._branch_cuts import (find_phase_residues_cy, branch_cut_dtype,
                           find_branch_cuts_cy, _prepare_branch_cuts_cy,
                           PERIODS_UNDEFINED, integrate_phase)
from .._shared.six import string_types
from ..morphology import label


def unwrap_phase(image, wrap_around=False, method=None):
    '''From ``image``, wrapped to lie in the interval [-pi, pi), recover the
    original, unwrapped image.

    Parameters
    ----------
    image : 1D, 2D or 3D ndarray of floats, optionally a masked array
        The values should be in the range ``[-pi, pi)``. If a masked array is
        provided, the masked entries will not be changed, and their values
        will not be used to guide the unwrapping of neighboring, unmasked
        values. Masked 1D arrays are not allowed, and will raise a
        ``ValueError``.
    wrap_around : bool or sequence of bool
        When an element of the sequence is  ``True``, the unwrapping process
        will regard the edges along the corresponding axis of the image to be
        connected and use this connectivity to guide the phase unwrapping
        process. If only a single boolean is given, it will apply to all axes.
        Wrap around is not supported for 1D arrays.
    method : string or ``None``
        Which unwrapping algorithm to use. One of ``None`` (default; will
        choose the default algorithm depending on the dimensionality of
        ``image``), ``'reliability'``, ``'branch_cuts'`` and ``'naive'``.
        Which algorithms are available depends on the dimensionality of
        ``image``, see below.

    Returns
    -------
    image_unwrapped : array_like, float
        Unwrapped image of the same shape as the input. If the input ``image``
        was a masked array, the mask will be preserved.

    Raises
    ------
    ValueError
        If called with a masked 1D array or called with a 1D array and
        ``wrap_around=True``.

    Notes
    -----
    For a nice introduction to phase unwrapping, see [1]_.

    Algorithms
    ----------
    This section gives an overview of the algorithms

    Naive (``method='naive'``; 1D only)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The default in 1D. This algorithm scans the array starting from ``image[0]``
    and proceeding towards higher indices, adding multiples of two pi to
    minimize the difference of neighboring pixels. Masked arrays are not
    supported, as masked data in 1D results in a problem with multiple
    solutions.

    Reliability (``method='reliability'``; 2D and 3D)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    The default in 2D and 3D. The algorithm is described by Gdeisat et al.
    in [2]_ and [3]_. Masked arrays are supported.

    Branch cuts (``method='branch_cuts'``; 2D only)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    This is the classical algorithm described by Goldstein [1]_, augmented to
    allow masked arrays.

    Examples
    --------
    >>> c0, c1 = np.ogrid[-1:1:128j, -1:1:128j]
    >>> image = 12 * np.pi * np.exp(-(c0**2 + c1**2))
    >>> image_wrapped = np.angle(np.exp(1j * image))
    >>> image_unwrapped = unwrap_phase(image_wrapped)
    >>> np.std(image_unwrapped - image) < 1e-6   # A constant offset is normal
    True

    References
    ----------
    .. [1] R. M. Goldstein, H. A. Zebker, C. L. Werner, "Satellite radar
           interferometry: Two-dimensional phase unwrapping", Radio Science 23
           (1988) 4, pp 713--720.
    .. [2] Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor,
           and Munther A. Gdeisat, "Fast two-dimensional phase-unwrapping
           algorithm based on sorting by reliability following a noncontinuous
           path", Journal Applied Optics, Vol. 41, No. 35 (2002) 7437,
    .. [3] Abdul-Rahman, H., Gdeisat, M., Burton, D., & Lalor, M., "Fast
           three-dimensional phase-unwrapping algorithm based on sorting by
           reliability following a non-continuous path. In W. Osten,
           C. Gorecki, & E. L. Novak (Eds.), Optical Metrology (2005) 32--40,
           International Society for Optics and Photonics.
    '''
    if image.ndim not in (1, 2, 3):
        raise ValueError('image must be 1, 2 or 3 dimensional')
    wrap_around = _normalize_wrap_around(wrap_around, image.ndim)
    if image.ndim > 1 and 1 in image.shape:
        warnings.warn('image has a length 1 dimension; consider using an '
                      'array of lower dimensionality to use a more efficient '
                      'algorithm')

    if method is None:
        if image.ndim == 1:
            method = 'naive'
        else:
            method = 'reliability'
    if method == 'naive':
        if wrap_around[0]:
            raise ValueError('wrap_around is not supported for 1D images')
        image_unwrapped = phase_unwrap_naive(image)
    elif method == 'reliability':
        image_unwrapped = unwrap_phase_reliability(image, wrap_around)
    elif method == 'branch_cuts':
        image_unwrapped = unwrap_phase_branch_cuts(image, wrap_around)
    else:
        raise ValueError('Unsupported method: %r' % method)

    if np.ma.isMaskedArray(image):
        image_unwrapped = np.ma.array(image_unwrapped, mask=image.mask)
    return image_unwrapped


def unwrap_phase_reliability(image, wrap_around=False):
    '''From ``image``, wrapped to lie in the interval [-pi, pi), recover the
    original, unwrapped image.

    Parameters
    ----------
    image : 2D or 3D ndarray of floats, optionally a masked array
        The values should be in the range ``[-pi, pi)``. If a masked array is
        provided, the masked entries will not be changed, and their values
        will not be used to guide the unwrapping of neighboring, unmasked
        values. Masked 1D arrays are not allowed, and will raise a
        ``ValueError``.
    wrap_around : bool or sequence of bool
        When an element of the sequence is  ``True``, the unwrapping process
        will regard the edges along the corresponding axis of the image to be
        connected and use this connectivity to guide the phase unwrapping
        process. If only a single boolean is given, it will apply to all axes.
        Wrap around is not supported for 1D arrays.

    Returns
    -------
    image_unwrapped : array_like, float
        Unwrapped image of the same shape as the input. If the input ``image``
        was a masked array, the mask will be preserved.

    Raises
    ------
    ValueError
        If called with a masked 1D array or called with a 1D array and
        ``wrap_around=True``.

    Examples
    --------
    >>> c0, c1 = np.ogrid[-1:1:128j, -1:1:128j]
    >>> image = 12 * np.pi * np.exp(-(c0**2 + c1**2))
    >>> image_wrapped = np.angle(np.exp(1j * image))
    >>> image_unwrapped = unwrap_phase(image_wrapped)
    >>> np.std(image_unwrapped - image) < 1e-6   # A constant offset is normal
    True

    References
    ----------
    .. [1] Miguel Arevallilo Herraez, David R. Burton, Michael J. Lalor,
           and Munther A. Gdeisat, "Fast two-dimensional phase-unwrapping
           algorithm based on sorting by reliability following a noncontinuous
           path", Journal Applied Optics, Vol. 41, No. 35 (2002) 7437,
    .. [2] Abdul-Rahman, H., Gdeisat, M., Burton, D., & Lalor, M., "Fast
           three-dimensional phase-unwrapping algorithm based on sorting by
           reliability following a non-continuous path. In W. Osten,
           C. Gorecki, & E. L. Novak (Eds.), Optical Metrology (2005) 32--40,
           International Society for Optics and Photonics.
    '''
    if image.ndim not in (2, 3):
        raise ValueError('image must be 2 or 3 dimensional')
    wrap_around = _normalize_wrap_around(wrap_around, image.ndim)

    if np.ma.isMaskedArray(image):
        mask = np.require(image.mask, np.uint8, ['C'])
    else:
        mask = np.zeros_like(image, dtype=np.uint8, order='C')
    image_not_masked = np.asarray(image, dtype=np.float64, order='C')
    image_unwrapped = np.empty_like(image, dtype=np.float64, order='C')

    if image.ndim == 2:
        unwrap_2d(image_not_masked, mask, image_unwrapped,
                  wrap_around)
    elif image.ndim == 3:
        unwrap_3d(image_not_masked, mask, image_unwrapped,
                  wrap_around)

    if np.ma.isMaskedArray(image):
        return np.ma.array(image_unwrapped, mask=mask)
    else:
        return image_unwrapped


def _normalize_wrap_around(wrap_around, ndim):
    if isinstance(wrap_around, bool):
        wrap_around = [wrap_around] * ndim
    elif (hasattr(wrap_around, '__getitem__')
          and not isinstance(wrap_around, string_types)):
        if len(wrap_around) != ndim:
            raise ValueError('Length of wrap_around must equal the '
                             'dimensionality of image')
        wrap_around = [bool(wa) for wa in wrap_around]
    else:
        raise ValueError('wrap_around must be a bool or a sequence with '
                         'length equal to the dimensionality of image')
    return wrap_around


def _prepare_branch_cuts(residues):
    branch_cuts = np.zeros(residues.shape, dtype=branch_cut_dtype, order='C')

    if not np.ma.isMaskedArray(residues):
        residues = np.ma.array(residues, mask=False)

    # Place edges between neighboring masked intersections
    # This also takes care of placing edges at the image border
    branch_cuts['vcut'][...] = (residues.mask
                                & np.roll(residues.mask, 1, axis=0))
    branch_cuts['hcut'][...] = (residues.mask
                                & np.roll(residues.mask, 1, axis=1))

    # Find regions of connected masked intersections
    labelled_mask = label(residues.mask, neighbors=4, background=0)
    number_of_masked_regions = np.max(labelled_mask) + 1

    # Allocate storage of residue values
    number_of_residues = np.sum(np.abs(residues))  # sum only unmasked entries
    # We will not store any residue in residue_storage[0]; therefore pad by 1
    residue_storage_size = number_of_residues + number_of_masked_regions + 1
    residue_storage = np.zeros((residue_storage_size,), dtype=np.int,
                               order='C')

    # Save residues in masked regions to the branch cut array
    residues_unmasked = np.asarray(residues)
    for l in range(number_of_masked_regions):
        region = labelled_mask == l
        residue = np.sum(residues_unmasked[region])
        residue_no = l + 1
        residue_storage[residue_no] = residue
        branch_cuts['residue_no'][region] = residue_no

    # Save normal (=not masked) residues to the branch cut array
    _prepare_branch_cuts_cy(branch_cuts,
                            residue_storage, number_of_masked_regions + 1,
                            np.asarray(residues, dtype=np.int8, order='C'),
                            np.require(residues.mask, np.uint8, ['C']))

    return branch_cuts, residue_storage


def find_branch_cuts(residues):
    '''Connect residues with branch cuts such that the length of the cuts
    is small subject to the constraint that the net residue of intersections
    along a cut should be zero.

    Parameters
    ----------
    residues : 2D ndarray of signed integers, optionally a masked array
        Residues to connect with branch cuts. The last row/column should
        correspond to residues wrapping around the image border. Both
        dimensions of the underlying image are assumed to wrap around;
        to achieve the effect of a border, mask all entries in the last
        row and column.

    Returns
    -------
    cut_vertical : 2D ndarray of uint8
        A value of ``1`` and ``0`` of element ``(i, j)`` indicates a cut
        and no cut, respectively, between pixels ``(i, j)`` and
        ``(i, j + 1)`` in the underlying image. Equivalently, the value of
        element ``(i, j)`` can be interpreted as the presence or absence
        of a cut between the intersections (potential residues) at
        ``(i, j)`` and ``(i - 1, j)``.
    cut_horizontal : 2D ndarray of uint8
        A value of ``1`` and ``0`` of element ``(i, j)`` indicates a cut
        and no cut, respectively, between pixels ``(i, j)`` and
        ``(i + 1, j)`` in the underlying image. Equivalently, the value of
        element ``(i, j)`` can be interpreted as the presence or absence
        of a cut between the intersections (potential residues) at
        ``(i, j)`` and ``(i, j-1)``.
    '''
    branch_cuts, residue_storage = _prepare_branch_cuts(residues)
    if np.ma.isMaskedArray(residues):
        residues_mask = np.require(residues.mask, np.uint8, ['C'])
    else:
        residues_mask = np.zeros_like(residues, dtype=np.uint8, order='C')
    branch_cuts = find_branch_cuts_cy(branch_cuts, residue_storage,
                                      residues_mask)
    return branch_cuts['vcut'], branch_cuts['hcut']


def find_phase_residues(image, wrap_around=False):
    '''Find the phase residues that will be encountered in phase unwrapping.

    Parameters
    ----------
    image : 2D ndarray of floats, optionally a masked array
        The values should be in the range ``[-pi, pi)``. The length of both
        dimensions must be at least 2.
    wrap_around : bool or sequence of bool
        When an element of the sequence is  ``True``, the residue search
        will regard the edges along the corresponding axis of the image to be
        connected. If only a single boolean is given, it will apply to both
        axes.

    Returns
    -------
    residues : 2D ndarray of integers, possibly a masked array
        Array where residues are indicated by non-zero values. The ``[0, 0]``
        element in ``residues`` corresponds to the loop ``[0, 0] -> [0, 1]
        -> [1, 1] -> [1, 0] -> [0, 0]`` in the ``image``. ``residues`` will
        have the same shape as ``image``, with the residues in the last
        row/column calculated by wrapping around the array. When
        ``wrap_around`` is ``False`` for an axis, the boundary elements will be
        masked. All intersections where one or more of the adjacent pixels
        were masked will also be masked.

    Notes
    -----
    The residues calculated here are defined as positive when traversing a
    clockwise path, matching the definition by Goldstein et al. [1]_.

    Examples
    --------
    This example is taken from [1]_. A positive residue is located in the
    center of ``image`` (in units of periods, the following path is the
    cause of the residue: ``0.0 -> 0.3 -> 0.6 -> 0.8 -> 0.0``).

    >>> image = 2 * np.pi * (np.array([[0.0, 0.1, 0.2, 0.3],
    ...                                [0.0, 0.0, 0.3, 0.4],
    ...                                [0.9, 0.8, 0.6, 0.5],
    ...                                [0.8, 0.8, 0.7, 0.6]]) - 0.5)
    >>> find_phase_residues(image)   # "--" indicates a masked element
    [[0 0 0 --]
     [0 1 0 --]
     [0 0 0 --]
     [-- -- -- --]]
    >>> find_phase_residues(image, wrap_around=True)
    [[ 0  0  0  0]
     [ 0  1  0 -1]
     [ 0  0  0  0]
     [ 0 -1  0  1]]

    References
    ----------
    .. [1] R. M. Goldstein, H. A. Zebker, C. L. Werner, "Satellite radar
           interferometry: Two-dimensional phase unwrapping", Radio Science 23
           (1988) 4, pp 713--720.
    '''
    if image.ndim != 2:
        raise ValueError('image must be 2D')
    wrap_around = _normalize_wrap_around(wrap_around, image.ndim)
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError('Residues cannot be determined for images with one '
                         'or more dimensions of length less than 2')

    # Calculate residues assuming wrap around on all axes and no image mask
    residues = find_phase_residues_cy(np.asarray(image, dtype=np.float64,
                                                 order='C'))

    # Account for axes without wrap around and image mask
    if (not all(wrap_around)) or np.ma.isMaskedArray(image):
        residues = np.ma.array(residues, mask=False)
    for i in range(image.ndim):
        # Roll border residues to last element along their axis
        residues = np.roll(residues, -1, axis=i)
        # Mask residues on boundaries that do not have wrap around
        if not wrap_around[i]:
            slice_ = tuple([-1 if j == i else slice(None)
                            for j in range(image.ndim)])
            residues.mask[slice_] = True
    if np.ma.isMaskedArray(image):
        # Mask residues that are calculated based on one or more masked pixels
        mask = (image.mask
                | np.roll(image.mask, 1, axis=0)
                | np.roll(image.mask, 1, axis=1)
                | np.roll(np.roll(image.mask, 1, axis=1), 1, axis=0))
        residues.mask |= mask

    return residues


def unwrap_phase_branch_cuts(image, wrap_around=False):
    '''From ``image``, wrapped to lie in the interval [-pi, pi), recover the
    original, unwrapped image using a method based on branch cuts [1]_.

    Parameters
    ----------
    image : 2D ndarray of floats, optionally a masked array
        The values should be in the range ``[-pi, pi)``. If a masked array is
        provided, the masked entries will not be changed, and their values
        will not be used to guide the unwrapping of neighboring, unmasked
        values. Masked 1D arrays are not allowed, and will raise a
        ``ValueError``.
    wrap_around : bool or sequence of bool
        When an element of the sequence is ``True``, the unwrapping process
        will regard the edges along the corresponding axis of the image to be
        connected and use this connectivity to guide the phase unwrapping
        process. If only a single boolean is given, it will apply to all axes.

    Returns
    -------
    image_unwrapped : array_like, float
        Unwrapped image of the same shape as the input. If the input ``image``
        was a masked array, the mask will be preserved.

    References
    ----------
    .. [1] R. M. Goldstein, H. A. Zebker, C. L. Werner, "Satellite radar
           interferometry: Two-dimensional phase unwrapping", Radio Science 23
           (1988) 4, pp 713--720.
    '''
    residues = find_phase_residues(image, wrap_around)
    cut_vertical, cut_horizontal = find_branch_cuts(residues)

    if np.ma.isMaskedArray(image):
        image_mask = np.require(image.mask, np.uint8, ['C'])
    else:
        image_mask = np.zeros_like(image, dtype=np.uint8, order='C')
    image_unmasked = np.asarray(image, dtype=np.float64, order='C')
    cut_vertical = np.require(cut_vertical, np.uint8, ['C'])
    cut_horizontal = np.require(cut_horizontal, np.uint8, ['C'])

    # Integrate phase
    periods = np.empty(image.shape, dtype=np.int64, order='C')
    periods[...] = PERIODS_UNDEFINED
    no_regions = 0
    remaining_pixels = (~image_mask) & (periods == PERIODS_UNDEFINED)
    while np.any(remaining_pixels):
        # Choose a start point (any remaining unmasked pixel)
        start_point = np.unravel_index(np.argmax(remaining_pixels), image.shape)
        periods = integrate_phase(image_unmasked, image_mask, periods,
                                  cut_vertical, cut_horizontal,
                                  start_point[0], start_point[1])
        no_regions += 1
        remaining_pixels = (~image_mask) & (periods == PERIODS_UNDEFINED)
    if no_regions > 1:
        warnings.warn('Unwrapped image was separated into %d regions by '
                      'residue cuts; these regions may have phase offsets '
                      'of n*2*pi, causing global errors' % no_regions)

    if np.ma.isMaskedArray(image):
        periods[periods == PERIODS_UNDEFINED] = 0
        image_unwrapped =  np.ma.array(image_unmasked + 2 * np.pi * periods,
                                       mask=image_mask)
    else:
        image_unwrapped = image_unmasked + 2 * np.pi * periods
    return image_unwrapped


def phase_unwrap_naive(image):
    '''Naive phase unwrapping of 1D arrays.

    Parameters
    ----------
    image : 1D ndarray of floats
        The values should be in the range ``[-pi, pi)``.

    Returns
    -------
    image_unwrapped : array_like, float64
        Unwrapped image of the same shape as the input.
    '''
    if image.ndim != 1:
        raise ValueError('image must be 1 dimensional')
    if np.ma.isMaskedArray(image):
        raise ValueError('1D masked images cannot be unwrapped')
    image_not_masked = np.asarray(image, dtype=np.float64, order='C')
    image_unwrapped = np.empty_like(image, dtype=np.float64, order='C')
    unwrap_naive_1d(image_not_masked, image_unwrapped)
    return image_unwrapped
