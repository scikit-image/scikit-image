from collections.abc import Sequence
from numbers import Integral

import numpy as np
from scipy import ndimage as ndi

from .. import draw
from .._shared.utils import deprecate_kwarg


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
    from skimage.morphology import binary_dilation

    # Create a single pixel image of sufficient size and apply binary dilation.
    shape = _shape_from_sequence(footprints)
    imag = np.zeros(shape, dtype=bool)
    imag[tuple(s // 2 for s in shape)] = 1
    return binary_dilation(imag, footprints)


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


@deprecate_kwarg({"height": "ncols", "width": "nrows"},
                 removed_version="0.20.0")
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


def disk(radius, dtype=np.uint8):
    """Generates a flat, disk-shaped footprint.

    A pixel is within the neighborhood if the Euclidean distance between
    it and the origin is no greater than radius.

    Parameters
    ----------
    radius : int
        The radius of the disk-shaped footprint.

    Other Parameters
    ----------------
    dtype : data-type, optional
        The data type of the footprint.

    Returns
    -------
    footprint : ndarray
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
    """
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)


def ellipse(width, height, dtype=np.uint8):
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

    Returns
    -------
    footprint : ndarray
        The footprint where elements of the neighborhood are 1 and 0 otherwise.

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
    footprint = np.zeros((2 * height + 1, 2 * width + 1), dtype=dtype)
    rows, cols = draw.ellipse(height, width, height + 1, width + 1)
    footprint[rows, cols] = 1
    return footprint


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


def ball(radius, dtype=np.uint8):
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

    Returns
    -------
    footprint : ndarray or tuple
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
    """
    n = 2 * radius + 1
    Z, Y, X = np.mgrid[-radius:radius:n * 1j,
                       -radius:radius:n * 1j,
                       -radius:radius:n * 1j]
    s = X ** 2 + Y ** 2 + Z ** 2
    return np.array(s <= radius * radius, dtype=dtype)


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


def _default_footprint(ndim):
    """Generates a cross-shaped footprint (connectivity=1).

    This is the default footprint (footprint) if no footprint was
    specified.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the image.

    Returns
    -------
    footprint : ndarray
        The footprint where elements of the neighborhood are 1 and 0 otherwise.

    """
    return ndi.generate_binary_structure(ndim, 1)
