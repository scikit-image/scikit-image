import os
from collections.abc import Sequence
from numbers import Integral

import numpy as np
from scipy import ndimage as ndi

from .. import draw
from .._shared.utils import deprecate_kwarg
from skimage import morphology

# Precomputed ball and disk decompositions were saved as 2D arrays where the
# radius of the desired decomposition is used to index into the first axis of
# the array. The values at a given radius corresponds to the number of
# repetitions of 3 different types elementary of structuring elements.
#
# See _nsphere_series_decomposition for full details.
_nsphere_decompositions = {}
_nsphere_decompositions[2] = np.load(
    os.path.join(os.path.dirname(__file__), 'disk_decompositions.npy'))
_nsphere_decompositions[3] = np.load(
    os.path.join(os.path.dirname(__file__), 'ball_decompositions.npy'))


def _footprint_is_sequence(footprint):
    if hasattr(footprint, '__array_interface__'):
        return False

    def _validate_sequence_element(t):
        return (
            isinstance(t, Sequence)
            and len(t) == 2
            and hasattr(t[0], '__array_interface__')
            and isinstance(t[1], Integral)
        )

    if isinstance(footprint, Sequence):
        if not all(_validate_sequence_element(t) for t in footprint):
            raise ValueError(
                "All elements of footprint sequence must be a 2-tuple where "
                "the first element of the tuple is an ndarray and the second "
                "is an integer indicating the number of iterations."
            )
    else:
        raise ValueError("footprint must be either an ndarray or Sequence")
    return True


def _shape_from_sequence(footprints, require_odd_size=False):
    """Determine the shape of composite footprint

    In the future if we only want to support odd-sized square, we may want to
    change this to require_odd_size
    """
    if not _footprint_is_sequence(footprints):
        raise ValueError("expected a sequence of footprints")
    ndim = footprints[0][0].ndim
    shape = [0] * ndim

    def _odd_size(size, require_odd_size):
        if require_odd_size and size % 2 == 0:
            raise ValueError(
                "expected all footprint elements to have odd size"
            )

    for d in range(ndim):
        fp, nreps = footprints[0]
        _odd_size(fp.shape[d], require_odd_size)
        shape[d] = fp.shape[d] + (nreps - 1) * (fp.shape[d] - 1)
        for fp, nreps in footprints[1:]:
            _odd_size(fp.shape[d], require_odd_size)
            shape[d] += nreps * (fp.shape[d] - 1)
    return tuple(shape)


def footprint_from_sequence(footprints):
    """Convert a footprint sequence into an equivalent ndarray.

    Parameters
    ----------
    footprints : tuple of 2-tuples
        A sequence of footprint tuples where the first element of each tuple
        is an array corresponding to a footprint and the second element is the
        number of times it is to be applied. Currently all footprints should
        have odd size.

    Returns
    -------
    footprint : ndarray
        An single array equivalent to applying the sequence of `footprints`.
    """

    # Create a single pixel image of sufficient size and apply binary dilation.
    shape = _shape_from_sequence(footprints)
    imag = np.zeros(shape, dtype=bool)
    imag[tuple(s // 2 for s in shape)] = 1
    return morphology.binary_dilation(imag, footprints)


def square(width, dtype=np.uint8, *, decomposition=None):
    """Generates a flat, square-shaped footprint.

    Every pixel along the perimeter has a chessboard distance
    no greater than radius (radius=floor(width/2)) pixels.

    Parameters
    ----------
    width : int
        The width and height of the square.

    Other Parameters
    ----------------
    dtype : data-type, optional
        The data type of the footprint.
    decomposition : {None, 'separable', 'sequence'}, optional
        If None, a single array is returned. For 'sequence', a tuple of smaller
        footprints is returned. Applying this series of smaller footprints will
        given an identical result to a single, larger footprint, but often with
        better computational performance. See Notes for more details.
        With 'separable', this function uses separable 1D footprints for each
        axis. Whether 'seqeunce' or 'separable' is computationally faster may
        be architecture-dependent.

    Returns
    -------
    footprint : ndarray or tuple
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
        When `decomposition` is None, this is just a numpy.ndarray. Otherwise,
        this will be a tuple whose length is equal to the number of unique
        structuring elements to apply (see Notes for more detail)

    Notes
    -----
    When `decomposition` is not None, each element of the `footprint`
    tuple is a 2-tuple of the form ``(ndarray, num_iter)`` that specifies a
    footprint array and the number of iterations it is to be applied.

    For binary morphology, using ``decomposition='sequence'`` or
    ``decomposition='separable'`` were observed to give better performance than
    ``decomposition=None``, with the magnitude of the performance increase
    rapidly increasing with footprint size. For grayscale morphology with
    square footprints, it is recommended to use ``decomposition=None`` since
    the internal SciPy functions that are called already have a fast
    implementation based on separable 1D sliding windows.

    The 'sequence' decomposition mode only supports odd valued `width`. If
    `width` is even, the sequence used will be identical to the 'separable'
    mode.
    """
    if decomposition is None:
        return np.ones((width, width), dtype=dtype)

    if decomposition == 'separable' or width % 2 == 0:
        sequence = [(np.ones((width, 1), dtype=dtype), 1),
                    (np.ones((1, width), dtype=dtype), 1)]
    elif decomposition == 'sequence':
        # only handles odd widths
        sequence = [(np.ones((3, 3), dtype=dtype), _decompose_size(width, 3))]
    else:
        raise ValueError(f"Unrecognized decomposition: {decomposition}")
    return tuple(sequence)


def _decompose_size(size, kernel_size=3):
    """Determine number of repeated iterations for a `kernel_size` kernel.

    Returns how many repeated morphology operations with an element of size
    `kernel_size` is equivalent to a morphology with a single kernel of size
    `n`.

    """
    if kernel_size % 2 != 1:
        raise ValueError("only odd length kernel_size is supported")
    return 1 + (size - kernel_size) // (kernel_size - 1)


@deprecate_kwarg({'height': 'ncols', 'width': 'nrows'},
                 deprecated_version='0.18.0',
                 removed_version='0.20.0')
def rectangle(nrows, ncols, dtype=np.uint8, *, decomposition=None):
    """Generates a flat, rectangular-shaped footprint.

    Every pixel in the rectangle generated for a given width and given height
    belongs to the neighborhood.

    Parameters
    ----------
    nrows : int
        The number of rows of the rectangle.
    ncols : int
        The number of columns of the rectangle.

    Other Parameters
    ----------------
    dtype : data-type, optional
        The data type of the footprint.
    decomposition : {None, 'separable', 'sequence'}, optional
        If None, a single array is returned. For 'sequence', a tuple of smaller
        footprints is returned. Applying this series of smaller footprints will
        given an identical result to a single, larger footprint, but often with
        better computational performance. See Notes for more details.
        With 'separable', this function uses separable 1D footprints for each
        axis. Whether 'seqeunce' or 'separable' is computationally faster may
        be architecture-dependent.

    Returns
    -------
    footprint : ndarray or tuple
        A footprint consisting only of ones, i.e. every pixel belongs to the
        neighborhood. When `decomposition` is None, this is just a
        numpy.ndarray. Otherwise, this will be a tuple whose length is equal to
        the number of unique structuring elements to apply (see Notes for more
        detail)

    Notes
    -----
    When `decomposition` is not None, each element of the `footprint`
    tuple is a 2-tuple of the form ``(ndarray, num_iter)`` that specifies a
    footprint array and the number of iterations it is to be applied.

    For binary morphology, using ``decomposition='sequence'``
    was observed to give better performance, with the magnitude of the
    performance increase rapidly increasing with footprint size. For grayscale
    morphology with rectangular footprints, it is recommended to use
    ``decomposition=None`` since the internal SciPy functions that are called
    already have a fast implementation based on separable 1D sliding windows.

    The `sequence` decomposition mode only supports odd valued `nrows` and
    `ncols`. If either `nrows` or `ncols` is even, the sequence used will be
    identical to ``decomposition='separable'``.

    - The use of ``width`` and ``height`` has been deprecated in
      version 0.18.0. Use ``nrows`` and ``ncols`` instead.
    """
    if decomposition is None:  # TODO: check optimal width setting here
        return np.ones((nrows, ncols), dtype=dtype)

    even_rows = nrows % 2 == 0
    even_cols = ncols % 2 == 0
    if decomposition == 'separable' or even_rows or even_cols:
        sequence = [(np.ones((nrows, 1), dtype=dtype), 1),
                    (np.ones((1, ncols), dtype=dtype), 1)]
    elif decomposition == 'sequence':
        # this branch only support odd nrows, ncols
        sq_size = 3
        sq_reps = _decompose_size(min(nrows, ncols), sq_size)
        sequence = [(np.ones((3, 3), dtype=dtype), sq_reps)]
        if nrows > ncols:
            nextra = nrows - ncols
            sequence.append(
                (np.ones((nextra + 1, 1), dtype=dtype), 1)
            )
        elif ncols > nrows:
            nextra = ncols - nrows
            sequence.append(
                (np.ones((1, nextra + 1), dtype=dtype), 1)
            )
    else:
        raise ValueError(f"Unrecognized decomposition: {decomposition}")
    return tuple(sequence)


def diamond(radius, dtype=np.uint8, *, decomposition=None):
    """Generates a flat, diamond-shaped footprint.

    A pixel is part of the neighborhood (i.e. labeled 1) if
    the city block/Manhattan distance between it and the center of
    the neighborhood is no greater than radius.

    Parameters
    ----------
    radius : int
        The radius of the diamond-shaped footprint.

    Other Parameters
    ----------------
    dtype : data-type, optional
        The data type of the footprint.
    decomposition : {None, 'sequence'}, optional
        If None, a single array is returned. For 'sequence', a tuple of smaller
        footprints is returned. Applying this series of smaller footprints will
        given an identical result to a single, larger footprint, but with
        better computational performance. See Notes for more details.

    Returns
    -------
    footprint : ndarray or tuple
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
        When `decomposition` is None, this is just a numpy.ndarray. Otherwise,
        this will be a tuple whose length is equal to the number of unique
        structuring elements to apply (see Notes for more detail)

    Notes
    -----
    When `decomposition` is not None, each element of the `footprint`
    tuple is a 2-tuple of the form ``(ndarray, num_iter)`` that specifies a
    footprint array and the number of iterations it is to be applied.

    For either binary or grayscale morphology, using
    ``decomposition='sequence'`` was observed to have a performance benefit,
    with the magnitude of the benefit increasing with increasing footprint
    size.

    """
    if decomposition is None:
        L = np.arange(0, radius * 2 + 1)
        I, J = np.meshgrid(L, L)
        footprint = np.array(np.abs(I - radius) + np.abs(J - radius) <= radius,
                             dtype=dtype)
    elif decomposition == 'sequence':
        fp = diamond(1, dtype=dtype, decomposition=None)
        nreps = _decompose_size(2 * radius + 1, fp.shape[0])
        footprint = ((fp, nreps),)
    else:
        raise ValueError(f"Unrecognized decomposition: {decomposition}")
    return footprint


def _nsphere_series_decomposition(radius, ndim, dtype=np.uint8):
    """Generate a sequence of footprints approximating an n-sphere.

    Morphological operations with an n-sphere (hypersphere) footprint can be
    approximated by applying a series of smaller footprints of extent 3 along
    each axis. Specific solutions for this are given in [1]_ for the case of
    2D disks with radius 2 through 10.

    Here we used n-dimensional extensions of the "square", "diamond" and
    "t-shaped" elements from that publication. All of these elementary elements
    have size ``(3,) * ndim``. We numerically computed the number of
    repetitions of each element that gives the closest match to the disk
    (in 2D) or ball (in 3D) computed with ``decomposition=None``.

    The approach can be extended to higher dimensions, but we have only stored
    results for 2D and 3D at this point.

    Empirically, the shapes at large radius approach a hexadecagon
    (16-sides [2]_) in 2D and a rhombicuboctahedron (26-faces, [3]_) in 3D.

    References
    ----------
    .. [1] Park, H and Chin R.T. Decomposition of structuring elements for
           optimal implementation of morphological operations. In Proceedings:
           1997 IEEE Workshop on Nonlinear Signal and Image Processing, London,
           UK.
           https://www.iwaenc.org/proceedings/1997/nsip97/pdf/scan/ns970226.pdf
    .. [2] https://en.wikipedia.org/wiki/Hexadecagon
    .. [3] https://en.wikipedia.org/wiki/Rhombicuboctahedron
    """

    if radius == 1:
        # for radius 1 just use the exact shape (3,) * ndim solution
        kwargs = dict(dtype=dtype, strict_radius=False, decomposition=None)
        if ndim == 2:
            return ((disk(1, **kwargs), 1),)
        elif ndim == 3:
            return ((ball(1, **kwargs), 1),)

    # load precomputed decompositions
    if ndim not in _nsphere_decompositions:
        raise ValueError(
            "sequence decompositions are only currently available for "
            "2d disks or 3d balls"
        )
    precomputed_decompositions = _nsphere_decompositions[ndim]
    max_radius = precomputed_decompositions.shape[0]
    if radius > max_radius:
        raise ValueError(
            f"precomputed {ndim}D decomposition unavailable for "
            f"radius > {max_radius}"
        )
    num_t_series, num_diamond, num_square = precomputed_decompositions[radius]

    sequence = []
    if num_t_series > 0:
        # shape (3, ) * ndim "T-shaped" footprints
        all_t = _t_shaped_element_series(ndim=ndim, dtype=dtype)
        [sequence.append((t, num_t_series)) for t in all_t]
    if num_diamond > 0:
        d = np.zeros((3,) * ndim, dtype=dtype)
        sl = [slice(1, 2)] * ndim
        for ax in range(ndim):
            sl[ax] = slice(None)
            d[tuple(sl)] = 1
            sl[ax] = slice(1, 2)
        sequence.append((d, num_diamond))
    if num_square > 0:
        sq = np.ones((3, ) * ndim, dtype=dtype)
        sequence.append((sq, num_square))
    return tuple(sequence)


def _t_shaped_element_series(ndim=2, dtype=np.uint8):
    """A series of T-shaped structuring elements.

    In the 2D case this is a T-shaped element and its rotation at multiples of
    90 degrees. This series is used in efficient decompositions of disks of
    various radius as published in [1]_.

    The generalization to the n-dimensional case can be performed by having the
    "top" of the T to extend in (ndim - 1) dimensions and then producing a
    series of rotations such that the bottom end of the T points along each of
    ``2 * ndim`` orthogonal directions.
    """
    if ndim == 2:
        # The n-dimensional case produces the same set of footprints, but
        # the 2D example is retained here for clarity.
        t0 = np.array([[1, 1, 1],
                       [0, 1, 0],
                       [0, 1, 0]], dtype=dtype)
        t90 = np.rot90(t0, 1)
        t180 = np.rot90(t0, 2)
        t270 = np.rot90(t0, 3)
        return t0, t90, t180, t270
    else:
        # ndimensional generalization of the 2D case above
        all_t = []
        for ax in range(ndim):
            for idx in [0, 2]:
                t = np.zeros((3,) * ndim, dtype=dtype)
                sl = [slice(None)] * ndim
                sl[ax] = slice(idx, idx + 1)
                t[tuple(sl)] = 1
                sl = [slice(1, 2)] * ndim
                sl[ax] = slice(None)
                t[tuple(sl)] = 1
                all_t.append(t)
    return tuple(all_t)


def disk(radius, dtype=np.uint8, *, strict_radius=True, decomposition=None):
    """Generates a flat, disk-shaped footprint.

    A pixel is within the neighborhood if the Euclidean distance between
    it and the origin is no greater than radius (This is only approximately
    True, when `decomposition == 'sequence'`).

    Parameters
    ----------
    radius : int
        The radius of the disk-shaped footprint.

    Other Parameters
    ----------------
    dtype : data-type, optional
        The data type of the footprint.
    strict_radius : bool, optional
        If False, extend the radius by 0.5. This allows the circle to expand
        further within a cube that remains of size ``2 * radius + 1`` along
        each axis. This parameter is ignored if decomposition is not None.
    decomposition : {None, 'sequence', 'crosses'}, optional
        If None, a single array is returned. For 'sequence', a tuple of smaller
        footprints is returned. Applying this series of smaller footprints will
        given a result equivalent to a single, larger footprint, but with
        better computational performance. For disk footprints, the 'sequence'
        or 'crosses' decompositions are not always exactly equivalent to
        ``decomposition=None``. See Notes for more details.

    Returns
    -------
    footprint : ndarray
        The footprint where elements of the neighborhood are 1 and 0 otherwise.

    Notes
    -----
    When `decomposition` is not None, each element of the `footprint`
    tuple is a 2-tuple of the form ``(ndarray, num_iter)`` that specifies a
    footprint array and the number of iterations it is to be applied.

    The disk produced by the ``decomposition='sequence'`` mode may not be
    identical to that with ``decomposition=None``. A disk footprint can be
    approximated by applying a series of smaller footprints of extent 3 along
    each axis. Specific solutions for this are given in [1]_ for the case of
    2D disks with radius 2 through 10. Here, we numerically computed the number
    of repetitions of each element that gives the closest match to the disk
    computed with kwargs ``strict_radius=False, decomposition=None``.

    Empirically, the series decomposition at large radius approaches a
    hexadecagon (a 16-sided polygon [2]_). In [3]_, the authors demonstrate
    that a hexadecagon is the closest approximation to a disk that can be
    achieved for decomposition with footprints of shape (3, 3).

    The disk produced by the ``decomposition='crosses'`` is often but not
    always  identical to that with ``decomposition=None``. It tends to give a
    closer approximation than ``decomposition='sequence'``, at a performance
    that is fairly comparable. The individual cross-shaped elements are not
    limited to extent (3, 3) in size. Unlike the 'seqeuence' decomposition, the
    'crosses' decomposition can also accurately approximate the shape of disks
    with ``strict_radius=True``. The method is based on an adaption of
    algorithm 1 given in [4]_.

    References
    ----------
    .. [1] Park, H and Chin R.T. Decomposition of structuring elements for
           optimal implementation of morphological operations. In Proceedings:
           1997 IEEE Workshop on Nonlinear Signal and Image Processing, London,
           UK.
           https://www.iwaenc.org/proceedings/1997/nsip97/pdf/scan/ns970226.pdf
    .. [2] https://en.wikipedia.org/wiki/Hexadecagon
    .. [3] Vanrell, M and Vitrià, J. Optimal 3 × 3 decomposable disks for
           morphological transformations. Image and Vision Computing, Vol. 15,
           Issue 11, 1997.
           :DOI:`10.1016/S0262-8856(97)00026-7`
    .. [4] Li, D. and Ritter, G.X. Decomposition of Separable and Symmetric
           Convex Templates. Proc. SPIE 1350, Image Algebra and Morphological
           Image Processing, (1 November 1990).
           :DOI:`10.1117/12.23608`
    """
    if decomposition is None:
        L = np.arange(-radius, radius + 1)
        X, Y = np.meshgrid(L, L)
        if not strict_radius:
            radius += 0.5
        return np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
    elif decomposition == 'sequence':
        sequence = _nsphere_series_decomposition(radius, ndim=2, dtype=dtype)
    elif decomposition == 'crosses':
        fp = disk(radius, dtype, strict_radius=strict_radius,
                  decomposition=None)
        sequence = _cross_decomposition(fp)
    return sequence


def _cross(r0, r1, dtype=np.uint8):
    """Cross-shaped structuring element of shape (r0, r1).

    Only the central row and column are ones.
    """
    s0 = int(2 * r0 + 1)
    s1 = int(2 * r1 + 1)
    c = np.zeros((s0, s1), dtype=dtype)
    if r1 != 0:
        c[r0, :] = 1
    if r0 != 0:
        c[:, r1] = 1
    return c


def _cross_decomposition(footprint, dtype=np.uint8):
    """ Decompose a symmetric convex footprint into cross-shaped elements.

    This is a decomposition of the footprint into a sequence of
    (possibly asymmetric) cross-shaped elements. This technique was proposed in
    [1]_ and corresponds roughly to algorithm 1 of that publication (some
    details had to be modified to get reliable operation).

    .. [1] Li, D. and Ritter, G.X. Decomposition of Separable and Symmetric
           Convex Templates. Proc. SPIE 1350, Image Algebra and Morphological
           Image Processing, (1 November 1990).
           :DOI:`10.1117/12.23608`
    """
    quadrant = footprint[footprint.shape[0] // 2:, footprint.shape[1] // 2:]
    col_sums = quadrant.sum(0, dtype=int)
    col_sums = np.concatenate((col_sums, np.asarray([0], dtype=int)))
    i_prev = 0
    idx = {}
    sum0 = 0
    for i in range(col_sums.size - 1):
        if col_sums[i] > col_sums[i + 1]:
            if i == 0:
                continue
            key = (col_sums[i_prev] - col_sums[i], i - i_prev)
            sum0 += key[0]
            if key not in idx:
                idx[key] = 1
            else:
                idx[key] += 1
            i_prev = i
    n = quadrant.shape[0] - 1 - sum0
    if n > 0:
        key = (n, 0)
        idx[key] = idx.get(key, 0) + 1
    return tuple([(_cross(r0, r1, dtype), n) for (r0, r1), n in idx.items()])


def ellipse(width, height, dtype=np.uint8, *, decomposition=None):
    """Generates a flat, ellipse-shaped footprint.

    Every pixel along the perimeter of ellipse satisfies
    the equation ``(x/width+1)**2 + (y/height+1)**2 = 1``.

    Parameters
    ----------
    width : int
        The width of the ellipse-shaped footprint.
    height : int
        The height of the ellipse-shaped footprint.

    Other Parameters
    ----------------
    dtype : data-type, optional
        The data type of the footprint.
    decomposition : {None, 'crosses'}, optional
        If None, a single array is returned. For 'sequence', a tuple of smaller
        footprints is returned. Applying this series of smaller footprints will
        given an identical result to a single, larger footprint, but with
        better computational performance. See Notes for more details.

    Returns
    -------
    footprint : ndarray
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
        The footprint will have shape ``(2 * height + 1, 2 * width + 1)``.

    Notes
    -----
    When `decomposition` is not None, each element of the `footprint`
    tuple is a 2-tuple of the form ``(ndarray, num_iter)`` that specifies a
    footprint array and the number of iterations it is to be applied.

    The ellipse produced by the ``decomposition='crosses'`` is often but not
    always  identical to that with ``decomposition=None``. The method is based
    on an adaption of algorithm 1 given in [1]_.

    References
    ----------
    .. [1] Li, D. and Ritter, G.X. Decomposition of Separable and Symmetric
           Convex Templates. Proc. SPIE 1350, Image Algebra and Morphological
           Image Processing, (1 November 1990).
           :DOI:`10.1117/12.23608`

    Examples
    --------
    >>> from skimage.morphology import footprints
    >>> footprints.ellipse(5, 3)
    array([[0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
           [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=uint8)

    """
    if decomposition is None:
        footprint = np.zeros((2 * height + 1, 2 * width + 1), dtype=dtype)
        rows, cols = draw.ellipse(height, width, height + 1, width + 1)
        footprint[rows, cols] = 1
        return footprint
    elif decomposition == 'crosses':
        fp = ellipse(width, height, dtype, decomposition=None)
        sequence = _cross_decomposition(fp)
    return sequence


def cube(width, dtype=np.uint8, *, decomposition=None):
    """ Generates a cube-shaped footprint.

    This is the 3D equivalent of a square.
    Every pixel along the perimeter has a chessboard distance
    no greater than radius (radius=floor(width/2)) pixels.

    Parameters
    ----------
    width : int
        The width, height and depth of the cube.

    Other Parameters
    ----------------
    dtype : data-type, optional
        The data type of the footprint.
    decomposition : {None, 'separable', 'sequence'}, optional
        If None, a single array is returned. For 'sequence', a tuple of smaller
        footprints is returned. Applying this series of smaller footprints will
        given an identical result to a single, larger footprint, but often with
        better computational performance. See Notes for more details.

    Returns
    -------
    footprint : ndarray or tuple
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
        When `decomposition` is None, this is just a numpy.ndarray. Otherwise,
        this will be a tuple whose length is equal to the number of unique
        structuring elements to apply (see Notes for more detail)

    Notes
    -----
    When `decomposition` is not None, each element of the `footprint`
    tuple is a 2-tuple of the form ``(ndarray, num_iter)`` that specifies a
    footprint array and the number of iterations it is to be applied.

    For binary morphology, using ``decomposition='sequence'``
    was observed to give better performance, with the magnitude of the
    performance increase rapidly increasing with footprint size. For grayscale
    morphology with square footprints, it is recommended to use
    ``decomposition=None`` since the internal SciPy functions that are called
    already have a fast implementation based on separable 1D sliding windows.

    The 'sequence' decomposition mode only supports odd valued `width`. If
    `width` is even, the sequence used will be identical to the 'separable'
    mode.
    """
    if decomposition is None:
        return np.ones((width, width, width), dtype=dtype)

    if decomposition == 'separable' or width % 2 == 0:
        sequence = [(np.ones((width, 1, 1), dtype=dtype), 1),
                    (np.ones((1, width, 1), dtype=dtype), 1),
                    (np.ones((1, 1, width), dtype=dtype), 1)]
    elif decomposition == 'sequence':
        # only handles odd widths
        sequence = [
            (np.ones((3, 3, 3), dtype=dtype), _decompose_size(width, 3))
        ]
    else:
        raise ValueError(f"Unrecognized decomposition: {decomposition}")
    return tuple(sequence)


def octahedron(radius, dtype=np.uint8, *, decomposition=None):
    """Generates a octahedron-shaped footprint.

    This is the 3D equivalent of a diamond.
    A pixel is part of the neighborhood (i.e. labeled 1) if
    the city block/Manhattan distance between it and the center of
    the neighborhood is no greater than radius.

    Parameters
    ----------
    radius : int
        The radius of the octahedron-shaped footprint.

    Other Parameters
    ----------------
    dtype : data-type, optional
        The data type of the footprint.
    decomposition : {None, 'sequence'}, optional
        If None, a single array is returned. For 'sequence', a tuple of smaller
        footprints is returned. Applying this series of smaller footprints will
        given an identical result to a single, larger footprint, but with
        better computational performance. See Notes for more details.

    Returns
    -------
    footprint : ndarray or tuple
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
        When `decomposition` is None, this is just a numpy.ndarray. Otherwise,
        this will be a tuple whose length is equal to the number of unique
        structuring elements to apply (see Notes for more detail)

    Notes
    -----
    When `decomposition` is not None, each element of the `footprint`
    tuple is a 2-tuple of the form ``(ndarray, num_iter)`` that specifies a
    footprint array and the number of iterations it is to be applied.

    For either binary or grayscale morphology, using
    ``decomposition='sequence'`` was observed to have a performance benefit,
    with the magnitude of the benefit increasing with increasing footprint
    size.
    """
    # note that in contrast to diamond(), this method allows non-integer radii
    if decomposition is None:
        n = 2 * radius + 1
        Z, Y, X = np.mgrid[-radius:radius:n * 1j,
                           -radius:radius:n * 1j,
                           -radius:radius:n * 1j]
        s = np.abs(X) + np.abs(Y) + np.abs(Z)
        footprint = np.array(s <= radius, dtype=dtype)
    elif decomposition == 'sequence':
        fp = octahedron(1, dtype=dtype, decomposition=None)
        nreps = _decompose_size(2 * radius + 1, fp.shape[0])
        footprint = ((fp, nreps),)
    else:
        raise ValueError(f"Unrecognized decomposition: {decomposition}")
    return footprint


def ball(radius, dtype=np.uint8, *, strict_radius=True, decomposition=None):
    """Generates a ball-shaped footprint.

    This is the 3D equivalent of a disk.
    A pixel is within the neighborhood if the Euclidean distance between
    it and the origin is no greater than radius.

    Parameters
    ----------
    radius : int
        The radius of the ball-shaped footprint.

    Other Parameters
    ----------------
    dtype : data-type, optional
        The data type of the footprint.
    strict_radius : bool, optional
        If False, extend the radius by 0.5. This allows the circle to expand
        further within a cube that remains of size ``2 * radius + 1`` along
        each axis. This parameter is ignored if decomposition is not None.
    decomposition : {None, 'sequence'}, optional
        If None, a single array is returned. For 'sequence', a tuple of smaller
        footprints is returned. Applying this series of smaller footprints will
        given a result equivalent to a single, larger footprint, but with
        better computational performance. For ball footprints, the sequence
        decomposition is not exactly equivalent to decomposition=None.
        See Notes for more details.

    Returns
    -------
    footprint : ndarray or tuple
        The footprint where elements of the neighborhood are 1 and 0 otherwise.

    Notes
    -----
    The disk produced by the decomposition='sequence' mode is not identical
    to that with decomposition=None. Here we extend the approach taken in [1]_
    for disks to the 3D case, using 3-dimensional extensions of the "square",
    "diamond" and "t-shaped" elements from that publication. All of these
    elementary elements have size ``(3,) * ndim``. We numerically computed the
    number of repetitions of each element that gives the closest match to the
    ball computed with kwargs ``strict_radius=False, decomposition=None``.

    Empirically, the equivalent composite footprint to the sequence
    decomposition approaches a rhombicuboctahedron (26-faces [2]_).

    References
    ----------
    .. [1] Park, H and Chin R.T. Decomposition of structuring elements for
           optimal implementation of morphological operations. In Proceedings:
           1997 IEEE Workshop on Nonlinear Signal and Image Processing, London,
           UK.
           https://www.iwaenc.org/proceedings/1997/nsip97/pdf/scan/ns970226.pdf
    .. [2] https://en.wikipedia.org/wiki/Rhombicuboctahedron
    """
    if decomposition is None:
        n = 2 * radius + 1
        Z, Y, X = np.mgrid[-radius:radius:n * 1j,
                           -radius:radius:n * 1j,
                           -radius:radius:n * 1j]
        s = X ** 2 + Y ** 2 + Z ** 2
        if not strict_radius:
            radius += 0.5
        return np.array(s <= radius * radius, dtype=dtype)
    elif decomposition == 'sequence':
        sequence = _nsphere_series_decomposition(radius, ndim=3, dtype=dtype)
    else:
        raise ValueError(f"Unrecognized decomposition: {decomposition}")
    return sequence


def octagon(m, n, dtype=np.uint8, *, decomposition=None):
    """Generates an octagon shaped footprint.

    For a given size of (m) horizontal and vertical sides
    and a given (n) height or width of slanted sides octagon is generated.
    The slanted sides are 45 or 135 degrees to the horizontal axis
    and hence the widths and heights are equal. The overall size of the
    footprint along a single axis will be ``m + 2 * n``.

    Parameters
    ----------
    m : int
        The size of the horizontal and vertical sides.
    n : int
        The height or width of the slanted sides.

    Other Parameters
    ----------------
    dtype : data-type, optional
        The data type of the footprint.
    decomposition : {None, 'sequence'}, optional
        If None, a single array is returned. For 'sequence', a tuple of smaller
        footprints is returned. Applying this series of smaller footprints will
        given an identical result to a single, larger footprint, but with
        better computational performance. See Notes for more details.

    Returns
    -------
    footprint : ndarray or tuple
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
        When `decomposition` is None, this is just a numpy.ndarray. Otherwise,
        this will be a tuple whose length is equal to the number of unique
        structuring elements to apply (see Notes for more detail)

    Notes
    -----
    When `decomposition` is not None, each element of the `footprint`
    tuple is a 2-tuple of the form ``(ndarray, num_iter)`` that specifies a
    footprint array and the number of iterations it is to be applied.

    For either binary or grayscale morphology, using
    ``decomposition='sequence'`` was observed to have a performance benefit,
    with the magnitude of the benefit increasing with increasing footprint
    size.
    """
    if m == n == 0:
        raise ValueError("m and n cannot both be zero")

    # TODO?: warn about even footprint size when m is even

    if decomposition is None:
        from . import convex_hull_image
        footprint = np.zeros((m + 2 * n, m + 2 * n))
        footprint[0, n] = 1
        footprint[n, 0] = 1
        footprint[0, m + n - 1] = 1
        footprint[m + n - 1, 0] = 1
        footprint[-1, n] = 1
        footprint[n, -1] = 1
        footprint[-1, m + n - 1] = 1
        footprint[m + n - 1, -1] = 1
        footprint = convex_hull_image(footprint).astype(dtype)
    elif decomposition == 'sequence':
        # special handling for edge cases with small m and/or n
        if m <= 2 and n <= 2:
            return ((octagon(m, n, dtype=dtype, decomposition=None), 1),)

        # general approach for larger m and/or n
        if m == 0:
            m = 2
            n -= 1
        sequence = []
        if m > 1:
            sequence += list(square(m, dtype=dtype, decomposition='sequence'))
        if n > 0:
            sequence += [(diamond(1, dtype=dtype, decomposition=None), n)]
        footprint = tuple(sequence)
    else:
        raise ValueError(f"Unrecognized decomposition: {decomposition}")
    return footprint


def star(a, dtype=np.uint8):
    """Generates a star shaped footprint.

    Start has 8 vertices and is an overlap of square of size `2*a + 1`
    with its 45 degree rotated version.
    The slanted sides are 45 or 135 degrees to the horizontal axis.

    Parameters
    ----------
    a : int
        Parameter deciding the size of the star structural element. The side
        of the square array returned is `2*a + 1 + 2*floor(a / 2)`.

    Other Parameters
    ----------------
    dtype : data-type, optional
        The data type of the footprint.

    Returns
    -------
    footprint : ndarray
        The footprint where elements of the neighborhood are 1 and 0 otherwise.

    """
    from . import convex_hull_image

    if a == 1:
        bfilter = np.zeros((3, 3), dtype)
        bfilter[:] = 1
        return bfilter

    m = 2 * a + 1
    n = a // 2
    footprint_square = np.zeros((m + 2 * n, m + 2 * n))
    footprint_square[n: m + n, n: m + n] = 1

    c = (m + 2 * n - 1) // 2
    footprint_rotated = np.zeros((m + 2 * n, m + 2 * n))
    footprint_rotated[0, c] = footprint_rotated[-1, c] = 1
    footprint_rotated[c, 0] = footprint_rotated[c, -1] = 1
    footprint_rotated = convex_hull_image(footprint_rotated).astype(int)

    footprint = footprint_square + footprint_rotated
    footprint[footprint > 0] = 1

    return footprint.astype(dtype)
