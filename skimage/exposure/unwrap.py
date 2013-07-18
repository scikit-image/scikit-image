import numpy as np
import warnings

from ._unwrap_1d import unwrap_1d
from ._unwrap_2d import unwrap_2d
from ._unwrap_3d import unwrap_3d
from .._shared.six import string_types
from ._goldstein import (find_phase_residues_cy, branch_cut_dtype,
                         find_branch_cuts_cy, _prepare_branch_cuts_cy)


def unwrap_phase(image, wrap_around=False):
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

    Returns
    -------
    image_unwrapped : array_like, float32
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
    if image.ndim not in (1, 2, 3):
        raise ValueError('image must be 1, 2 or 3 dimensional')
    if isinstance(wrap_around, bool):
        wrap_around = [wrap_around] * image.ndim
    elif (hasattr(wrap_around, '__getitem__')
          and not isinstance(wrap_around, string_types)):
        if len(wrap_around) != image.ndim:
            raise ValueError('Length of wrap_around must equal the '
                             'dimensionality of image')
        wrap_around = [bool(wa) for wa in wrap_around]
    else:
        raise ValueError('wrap_around must be a bool or a sequence with '
                         'length equal to the dimensionality of image')
    if image.ndim == 1:
        if np.ma.isMaskedArray(image):
            raise ValueError('1D masked images cannot be unwrapped')
        if wrap_around[0]:
            raise ValueError('wrap_around is not supported for 1D images')
    if image.ndim in (2, 3) and 1 in image.shape:
        warnings.warn('image has a length 1 dimension; consider using an '
                      'array of lower dimensionality to use a more efficient '
                      'algorithm')

    if np.ma.isMaskedArray(image):
        mask = np.require(image.mask, np.uint8, ['C'])
    else:
        mask = np.zeros_like(image, dtype=np.uint8, order='C')
    image_not_masked = np.asarray(image, dtype=np.float64, order='C')
    image_unwrapped = np.empty_like(image, dtype=np.float64, order='C')

    if image.ndim == 1:
        unwrap_1d(image_not_masked, image_unwrapped)
    elif image.ndim == 2:
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


def find_phase_residues(image, wrap_around=False):
    '''Find the phase residues that will be encountered in phase unwrapping.

    Parameters
    ----------
    image : 2D ndarray of floats
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
        masked.

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
    if np.ma.isMaskedArray(image):
        raise ValueError('Residues cannot be computed for masked arrays')
    wrap_around = _normalize_wrap_around(wrap_around, image.ndim)
    if image.shape[0] < 2 or image.shape[1] < 2:
        raise ValueError('Residues cannot be determined for images with one '
                         'or more dimensions of length less than 2')
    residues = find_phase_residues_cy(np.require(image, np.float64, ['C']))
    # Roll border residues to last element along their axis and mask
    # boundary residues for wrap_around=False
    if not all(wrap_around):
        residues = np.ma.array(residues, mask=False)
    for i in range(image.ndim):
        residues = np.roll(residues, -1, axis=i)
        if not wrap_around[i]:
            slice_ = tuple([-1 if j == i else slice(None)
                            for j in range(image.ndim)])
            residues.mask[slice_] = True

    return residues


def _find_branch_cuts(residues, mask=None, wrap_around=False):
    wrap_around = _normalize_wrap_around(wrap_around, residues.ndim)

    shape = tuple([s if wa else s + 1
                   for s, wa in zip(residues.shape, wrap_around)])
    if not all(wrap_around):
        raise NotImplementedError('Branch cuts without wrap around is not '
                                  'implemented yet')
    if not mask is None:
        raise NotImplementedError('Branch cuts with mask images is not '
                                  'implemented yet')
    branch_cuts = np.zeros(shape, dtype=branch_cut_dtype, order='C')

    # Place cuts at the border where there is no wrap around
    if not wrap_around[0]:
        branch_cuts['vcut'][-1, :] = 1
    if not wrap_around[1]:
        branch_cuts['hcut'][:, -1] = 1
    # TODO: How to treat the edge with wrap_around=False
    # Pad image; include the edge as a masked region

    # TODO: Place edges around masked regions
    # logical or of the mask and the mask shifted by (-1, -1) gives the
    # edges that should be set

    # TODO: Sum residues in masked regions and save them to the masked regions
    # Place residues in the residue array
    number_of_residues = np.sum(np.abs(residues))  # sum only unmasked entries
    # labelled_mask = Label the masked regions
    number_of_masked_regions = 0   # TODO
    # We will not store any residue in residue_storage[0]; therefore pad by 1
    residue_storage_size = number_of_residues + number_of_masked_regions + 1
    residue_storage = np.zeros((residue_storage_size,), dtype=np.int,
                               order='C')
    #for label in labels:
        #residue = sum of residues in and on the boundary of region
        #residue_storage[label] = residue
        #branch_cuts['residue_no'][labelled_mask] = label
    # Save normal residues to the branch cut array
    if np.ma.isMaskedArray(residues):
        residues_mask = np.require(residues_mask, np.uint8, ['C'])
    else:
        residues_mask = np.zeros(residues.shape, dtype=np.uint8, order='C')
    _prepare_branch_cuts(branch_cuts,
                         residue_storage, number_of_masked_regions + 1,
                         np.asarray(residues, dtype=np.int8, order='C'),
                         residues_mask)
    print(branch_cuts)
    print(residue_storage)
    find_branch_cuts(branch_cuts, residue_storage, residues_mask)
    print(branch_cuts)
    print(residues)
    return branch_cuts


def _unwrap_phase_goldstein(image, wrap_around=False):
    if not all(_normalize_wrap_around(wrap_around, image.ndim)):
        raise NotImplementedError
    residues = find_phase_residues(image, wrap_around)
    branch_cuts = _find_branch_cuts(residues, wrap_around=wrap_around)
    print(image / (2 * np.pi) + 0.5)

    return image
