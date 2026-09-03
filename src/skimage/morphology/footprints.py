import numpy as np

from _skimage2.morphology.footprints import (
    cross_decompose_footprint as _ski2_cross_decompose_footprint,
    diamond as diamond,
    footprint_disk_decomposed as _ski2_footprint_disk_decomposed,
    footprint_from_sequence as _ski2_footprint_from_sequence,
    footprint_rectangle_decomposed as _sk2_footprint_rectangle_decomposed,
    octagon as octagon,
    octahedron as octahedron,
    star as star,
)  # noqa: F401
from _skimage2.morphology._footprints import mirror_footprint, pad_footprint  # noqa: F401

from .._migration import ski2_migration_decorator
from ..draw.draw import ellipse as _draw_ellipse

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(
    globals(),
    skip_names=(
        "_sk2_footprint_rectangle_decomposed",
        "_ski2_footprint_disk_decomposed",
        "_ski2_cross_decompose_footprint",
        "_ski2_footprint_from_sequence",
    ),
)


__all__ = [
    'ball',
    'cube',
    'diamond',
    'disk',
    'ellipse',
    'footprint_from_sequence',
    'footprint_rectangle',
    'octagon',
    'octahedron',
    'rectangle',
    'square',
    'star',
]


@ski2_migration_decorator(
    """\
`skimage.morphology.footprint_from_sequence` is deprecated in favor of
`skimage2.morphology.footprint_from_sequence` with new default behavior.

By default, `skimage2.morphology.footprint_from_sequence` now returns the same
dtype of the first array in the given ``decomposed`` sequence
(`skimage.morphology.footprint_from_sequence` always returned ``dtype=bool``).

To keep the old behavior when switching to `skimage2`, use the new parameter
and pass ``dtype=bool`` to the function.

<!--- cond-start: doc -->

>>> import numpy as np
>>> import skimage as ski1
>>> import skimage2 as ski2

>>> decomposed = (
...     (np.ones((3, 1), dtype=np.uint8), 1),
...     (np.ones((1, 3), dtype=np.uint8), 2),
... )

>>> fp1 = ski1.morphology.footprint_from_sequence(decomposed)
>>> fp2 = ski2.morphology.footprint_from_sequence(decomposed, dtype=bool)
>>> assert fp1.dtype == fp2.dtype
>>> np.testing.assert_equal(fp1, fp2)

<!--- cond-end -->
""",
    qname_old="skimage.morphology.footprint_from_sequence",
)
def footprint_from_sequence(footprints):
    """Convert a footprint sequence into an equivalent ndarray.

    Parameters
    ----------
    footprints : tuple of 2-tuples
        A sequence of footprint tuples where the first element of each tuple
        is an array corresponding to a footprint and the second element is the
        number of times it is to be applied. Currently, all footprints should
        have odd size.

    Returns
    -------
    footprint : ndarray
        An single array equivalent to applying the sequence of ``footprints``.
    """
    return _ski2_footprint_from_sequence(footprints, dtype=bool)


@ski2_migration_decorator(
    """\
`%(qname_old)s` is deprecated in favor of
`skimage2.morphology.footprint_rectangle` and
`skimage2.morphology.footprint_rectangle_decomposed`.

* `skimage2.morphology.footprint_rectangle` no longer accepts the ``decompose``
  parameter and will return the footprint as a simple array.
* `skimage2.morphology.footprint_rectangle_decomposed` uses the new parameter
  ``method``, which accepts the values of the old ``decomposition`` parameter.

To keep the old behavior when switching to `skimage2`, update your call
according to the following cases:

* ``decomposition`` not passed, use `skimage2.morphology.footprint_rectangle`
  with same signature.
* ``decomposition='sequence'`` or ``decomposition='separable'``, use
  `skimage2.morphology.footprint_rectangle_decomposed` and pass the old value
  of ``decomposition`` to the new parameter ``method``.

Other keyword parameters can be left unchanged.

<!--- cond-start: doc -->

>>> import numpy as np
>>> import skimage as ski1
>>> import skimage2 as ski2

>>> fp1 = ski1.morphology.%(qual)s((3, 3, 3))
>>> fp2 = ski2.morphology.footprint_rectangle((3, 3, 3))
>>> np.testing.assert_equal(fp1, fp2)

>>> fp1 = ski1.morphology.%(qual)s(
...     (3, 3, 3), decomposition="sequence"
... )
>>> fp2 = ski2.morphology.footprint_rectangle_decomposed(
...     (3, 3, 3), method="sequence"
... )
>>> np.testing.assert_equal(fp1, fp2)

<!--- cond-end -->
""",
    qname_old="skimage.morphology.footprint_rectangle",
)
def footprint_rectangle(shape, *, dtype=np.uint8, decomposition=None):
    """Generate a rectangular or hyper-rectangular footprint.

    Generates, depending on the length and dimensions requested with `shape`,
    a square, rectangle, cube, cuboid, or even higher-dimensional versions
    of these shapes.

    Parameters
    ----------
    shape : tuple[int, ...]
        The length of the footprint in each dimension. The length of the
        sequence determines the number of dimensions of the footprint.
    dtype : dtype-like, optional
        The data type of the footprint.
    decomposition : {None, 'separable', 'sequence'}, optional
        If None, a single array is returned. For 'sequence', a tuple of smaller
        footprints is returned. Applying this series of smaller footprints will
        give an identical result to a single, larger footprint, but often with
        better computational performance. See Notes for more details.
        With 'separable', this function uses separable 1D footprints for each
        axis. Whether 'sequence' or 'separable' is computationally faster may
        be architecture-dependent.

    Returns
    -------
    footprint : array or tuple[tuple[ndarray, int], ...]
        A footprint consisting only of ones, i.e. every pixel belongs to the
        neighborhood. When `decomposition` is None, this is just an array.
        Otherwise, this will be a tuple whose length is equal to the number of
        unique structuring elements to apply (see Examples for more detail).

    Examples
    --------
    >>> import skimage as ski
    >>> ski.morphology.footprint_rectangle((3, 5))
    array([[1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1],
           [1, 1, 1, 1, 1]], dtype=uint8)

    Decomposition will return multiple footprints that combine into a simple
    footprint of the requested shape.

    >>> ski.morphology.footprint_rectangle((9, 9), decomposition="sequence")
    ((array([[1, 1, 1],
             [1, 1, 1],
             [1, 1, 1]], dtype=uint8),
      4),)

    `"sequence"` makes sure that the decomposition only returns 1D footprints.

    >>> ski.morphology.footprint_rectangle((3, 5), decomposition="separable")
    ((array([[1],
             [1],
             [1]], dtype=uint8),
      1),
     (array([[1, 1, 1, 1, 1]], dtype=uint8), 1))

    Generate a 5-dimensional hypercube with 3 samples in each dimension

    >>> ski.morphology.footprint_rectangle((3,) * 5).shape
    (3, 3, 3, 3, 3)
    """
    if decomposition is None:
        footprint = np.ones(shape, dtype=dtype)
    else:
        footprint = _sk2_footprint_rectangle_decomposed(
            shape=shape,
            dtype=dtype,
            method=decomposition,
        )
    return footprint


@ski2_migration_decorator(
    """\
`%(qname_old)s` is deprecated in favor of
`skimage2.morphology.footprint_rectangle` and
`skimage2.morphology.footprint_rectangle_decomposed`.

* The new functions expect to be given a ``shape`` instead of the parameters
  ``nrows`` and ``ncols``.
* `skimage2.morphology.footprint_rectangle` no longer accepts the ``decompose``
  parameter and will return the footprint as a simple array.
* `skimage2.morphology.footprint_rectangle_decomposed` uses the new parameter
  ``method``, which accepts the values of the old ``decomposition`` parameter.

To keep the old behavior when switching to `skimage2`, update your call
according to the following cases:

* Pass the desired shape of the footprint as a 2-element tuple
  ``(nrows, ncols)`` of the formerly used parameters.
* ``decomposition`` not passed, use `skimage2.morphology.footprint_rectangle`
  with same signature.
* ``decomposition='sequence'`` or ``decomposition='separable'``, use
  `skimage2.morphology.footprint_rectangle_decomposed` and pass the old value
  of ``decomposition`` to the new parameter ``method``.

Other keyword parameters can be left unchanged.

<!--- cond-start: doc -->

>>> import numpy as np
>>> import skimage as ski1
>>> import skimage2 as ski2

>>> fp1 = ski1.morphology.%(qual)s(3, 3)
>>> fp2 = ski2.morphology.footprint_rectangle((3, 3))
>>> np.testing.assert_equal(fp1, fp2)

>>> fp1 = ski1.morphology.%(qual)s(
...     3, 3, decomposition="sequence"
... )
>>> fp2 = ski2.morphology.footprint_rectangle_decomposed(
...     (3, 3), method="sequence"
... )
>>> np.testing.assert_equal(fp1, fp2)

<!--- cond-end -->
""",
    qname_old="skimage.morphology.rectangle",
)
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
    dtype : dtype-like, optional
        The data type of the footprint.
    decomposition : {None, 'separable', 'sequence'}, optional
        If None, a single array is returned. For 'sequence', a tuple of smaller
        footprints is returned. Applying this series of smaller footprints will
        given an identical result to a single, larger footprint, but often with
        better computational performance. See Notes for more details.
        With 'separable', this function uses separable 1D footprints for each
        axis. Whether 'sequence' or 'separable' is computationally faster may
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
    footprint = footprint_rectangle(
        shape=(nrows, ncols), dtype=dtype, decomposition=decomposition
    )
    return footprint


@ski2_migration_decorator(
    """\
`%(qname_old)s` is deprecated in favor of
`skimage2.morphology.footprint_rectangle` and
`skimage2.morphology.footprint_rectangle_decomposed`.

* The new functions expect to be given a ``shape`` instead of the parameter
  ``width``.
* `skimage2.morphology.footprint_rectangle` no longer accepts the ``decompose``
  parameter and will return the footprint as a simple array.
* `skimage2.morphology.footprint_rectangle_decomposed` uses the new parameter
  ``method``, which accepts the values of the old ``decomposition`` parameter.

To keep the old behavior when switching to `skimage2`, update your call
according to the following cases:

* Pass the desired shape of the footprint as a 2-element tuple ``(w, w)``
  where ``w`` is the former ``width`` that was used.
* ``decomposition`` not passed, use `skimage2.morphology.footprint_rectangle`
  with same signature.
* ``decomposition='sequence'`` or ``decomposition='separable'``, use
  `skimage2.morphology.footprint_rectangle_decomposed` and pass the old value
  of ``decomposition`` to the new parameter ``method``.

Other keyword parameters can be left unchanged.

<!--- cond-start: doc -->

>>> import numpy as np
>>> import skimage as ski1
>>> import skimage2 as ski2

>>> fp1 = ski1.morphology.%(qual)s(3)
>>> fp2 = ski2.morphology.footprint_rectangle((3, 3))
>>> np.testing.assert_equal(fp1, fp2)

>>> fp1 = ski1.morphology.%(qual)s(
...     3, decomposition="sequence"
... )
>>> fp2 = ski2.morphology.footprint_rectangle_decomposed(
...     (3, 3), method="sequence"
... )
>>> np.testing.assert_equal(fp1, fp2)

<!--- cond-end -->
""",
    qname_old="skimage.morphology.square",
)
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
    dtype : dtype-like, optional
        The data type of the footprint.
    decomposition : {None, 'separable', 'sequence'}, optional
        If None, a single array is returned. For 'sequence', a tuple of smaller
        footprints is returned. Applying this series of smaller footprints will
        give an identical result to a single, larger footprint, but often with
        better computational performance. See Notes for more details.
        With 'separable', this function uses separable 1D footprints for each
        axis. Whether 'sequence' or 'separable' is computationally faster may
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
    footprint = footprint_rectangle(
        shape=(width, width), dtype=dtype, decomposition=decomposition
    )
    return footprint


@ski2_migration_decorator(
    """\
`%(qname_old)s` is deprecated in favor of
`skimage2.morphology.footprint_rectangle` and
`skimage2.morphology.footprint_rectangle_decomposed`.

* The new functions expect to be given a ``shape`` instead of the parameter
  ``width``.
* `skimage2.morphology.footprint_rectangle` no longer accepts the ``decompose``
  parameter and will return the footprint as a simple array.
* `skimage2.morphology.footprint_rectangle_decomposed` uses the new parameter
  ``method``, which accepts the values of the old ``decomposition`` parameter.

To keep the old behavior when switching to `skimage2`, update your call
according to the following cases:

* Pass the desired shape of the footprint as a 3-element tuple ``(w, w, w)``
  where ``w`` is the former ``width`` that was used.
* ``decomposition`` not passed, use `skimage2.morphology.footprint_rectangle`
  with same signature.
* ``decomposition='sequence'`` or ``decomposition='separable'``, use
  `skimage2.morphology.footprint_rectangle_decomposed` and pass the old value
  of ``decomposition`` to the new parameter ``method``.

Other keyword parameters can be left unchanged.

<!--- cond-start: doc -->

>>> import numpy as np
>>> import skimage as ski1
>>> import skimage2 as ski2

>>> fp1 = ski1.morphology.%(qual)s(3)
>>> fp2 = ski2.morphology.footprint_rectangle((3, 3, 3))
>>> np.testing.assert_equal(fp1, fp2)

>>> fp1 = ski1.morphology.%(qual)s(
...     3, decomposition="sequence"
... )
>>> fp2 = ski2.morphology.footprint_rectangle_decomposed(
...     (3, 3, 3), method="sequence"
... )
>>> np.testing.assert_equal(fp1, fp2)

<!--- cond-end -->
""",
    qname_old="skimage.morphology.cube",
)
def cube(width, dtype=np.uint8, *, decomposition=None):
    """Generates a cube-shaped footprint.

    This is the 3D equivalent of a square.
    Every pixel along the perimeter has a chessboard distance
    no greater than radius (radius=floor(width/2)) pixels.

    Parameters
    ----------
    width : int
        The width, height and depth of the cube.

    Other Parameters
    ----------------
    dtype : dtype-like, optional
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
    footprint = footprint_rectangle(
        shape=(width, width, width), dtype=dtype, decomposition=decomposition
    )
    return footprint


@ski2_migration_decorator(
    """\
    ``%(qname_old)s`` is deprecated in favor of
    ``%(qname_new)s`` with a new signature and behavior:

    * ``width`` and ``height`` are replaced by the parameter ``radii``
    * ``decomposition`` parameter is removed in favor of new function
      `cross_decompose_footprint`

    WARNING: The new underlying algorithm is slightly different and compounding
    float errors may lead to a few pixels at the footprints edge being 0
    instead of 1. If you need pixel-perfect compatibility try tweaking
    ``adjust_edge`` slightly (like ±0.001) or check out the full migration guide
    for a function to vendor.

    To keep the old (``skimage``, v1.x) behavior after switching to
    ``skimage2``, pass the following parameters to the new function:

    * Instead of ``width`` and ``height``, use ``radii=(height, width)``
    * Use ``adjust_edge=0.9999``

    If you used ``decomposition='crosses'``, apply the new function
    `skimage2.morphology.cross_decompose_footprint` to the generated footprint.

    <!--- cond-start: doc -->
    For example:

    >>> import numpy as np
    >>> import skimage as ski
    >>> import _skimage2 as ski2

    >>> width = 4
    >>> height = 9

    >>> fp1 = ski.morphology.ellipse(width, height)
    >>> fp2 = ski2.morphology.footprint_ellipse(
    ...     (height, width), adjust_edge=0.9999
    ... )
    >>> np.testing.assert_equal(fp1, fp2)

    >>> fp1_decomp = ski.morphology.ellipse(width, height, decomposition='crosses')
    >>> fp2_decomp = ski2.morphology.cross_decompose_footprint(fp2)
    >>> np.testing.assert_equal(fp1_decomp, fp2_decomp)
    <!--- cond-end -->

    Other keyword parameters can be left unchanged.
    """,
    qname_old="skimage.morphology.ellipse",
    qname_new="skimage.morphology.footprint_ellipse",
)
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
    dtype : dtype-like, optional
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
        # Intentionally left this old implementation in place so that
        # compatibility tests comparing skimage and skimage2 behavior are
        # meaningful
        footprint = np.zeros((2 * height + 1, 2 * width + 1), dtype=dtype)
        rows, cols = _draw_ellipse(height, width, height + 1, width + 1)
        footprint[rows, cols] = 1
        return footprint
    elif decomposition == 'crosses':
        fp = ellipse(width, height, dtype, decomposition=None)
        sequence = _ski2_cross_decompose_footprint(fp)
    return sequence


@ski2_migration_decorator(
    """\
    ``%(qname_old)s`` is deprecated in favor of
    ``%(qname_new)s`` with a new signature and behavior:

    * ``radius`` and ``strict_radius`` are replaced by
      the parameters ``radii`` and ``adjust_edge``
    * ``decomposition`` parameter is removed in favor of new functions
      `cross_decompose_footprint` and `footprint_disk_decomposed`

    WARNING: The new underlying algorithm is slightly different and compounding
    float errors may lead to a few pixels at the footprints edge being 0
    instead of 1. If you need pixel-perfect compatibility try tweaking
    ``adjust_edge`` slightly (like ±0.001) or check out the full migration guide
    for a function to vendor.

    To *approximate* the old (``skimage``, v1.x) behavior after switching to
    ``skimage2``:

    * Instead of ``radius``, use ``radii=(radius, radius)``
    * Instead of ``strict_radius=True`` (default), use ``adjust_edge=0.001``
    * Instead of ``strict_radius=False``, use ``adjust_edge=0.5``

    If you used ``decomposition='crosses'``, apply the new function
    `skimage2.morphology.cross_decompose_footprint` to the generated footprint.
    If you used ``decomposition='sequence'``, instead use
    `skimage2.morphology.footprint_disk_decomposed` to generate the decomposed
    footprint.

    <!--- cond-start: doc -->
    For example:

    >>> import numpy as np
    >>> import skimage as ski
    >>> import _skimage2 as ski2
    >>> radius = 7

    >>> fp1_strict = ski.morphology.disk(radius)
    >>> fp2_strict = ski2.morphology.footprint_ellipse(radius, adjust_edge=0.001)
    >>> np.testing.assert_equal(fp1_strict, fp2_strict)

    >>> fp1 = ski.morphology.disk(radius, strict_radius=False)
    >>> fp2 = ski2.morphology.footprint_ellipse(radius, adjust_edge=0.5)
    >>> np.testing.assert_equal(fp1, fp2)

    >>> fp1_crosses = ski.morphology.disk(radius, decomposition="crosses")
    >>> fp2_crosses = ski2.morphology.cross_decompose_footprint(fp2_strict)
    >>> np.testing.assert_equal(fp1_crosses, fp2_crosses)

    >>> fp1_sequence = ski.morphology.disk(radius, decomposition="sequence")
    >>> fp2_sequence = ski2.morphology.footprint_disk_decomposed(radius)
    >>> np.testing.assert_equal(fp1_sequence, fp2_sequence)

    .. admonition:: Reproduce exact results of old implementation
        :class: note dropdown

        Use/vendor the following function:

        .. code:: python

            def disk(radius, strict_radius=True):
                L = np.arange(-radius, radius + 1)
                X, Y = np.meshgrid(L, L)
                if not strict_radius:
                    radius += 0.5
                return np.array((X**2 + Y**2) <= radius**2, dtype=dtype)

    <!--- cond-end -->

    Other keyword parameters can be left unchanged.
    """,
    qname_old="skimage.morphology.disk",
    qname_new="skimage.morphology.footpint_ellipse",
)
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
    dtype : dtype-like, optional
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
        # Intentionally left this old implementation in place so that
        # compatibility tests comparing skimage and skimage2 behavior are
        # meaningful
        L = np.arange(-radius, radius + 1)
        X, Y = np.meshgrid(L, L)
        if not strict_radius:
            radius += 0.5
        return np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
    elif decomposition == 'sequence':
        sequence = _ski2_footprint_disk_decomposed(radius, ndim=2, dtype=dtype)
    elif decomposition == 'crosses':
        fp = disk(radius, dtype, strict_radius=strict_radius, decomposition=None)
        sequence = _ski2_cross_decompose_footprint(fp)
    return sequence


@ski2_migration_decorator(
    """\
    ``%(qname_old)s`` is deprecated in favor of
    ``%(qname_new)s`` with a new signature and behavior:

    * ``radius`` and ``strict_radius`` are replaced by
      the parameters ``radii`` and ``adjust_edge``
    * ``decomposition`` parameter is removed

    WARNING: The new underlying algorithm is slightly different and compounding
    float errors may lead to a few pixels at the footprints edge being 0
    instead of 1. If you need pixel-perfect compatibility try tweaking
    ``adjust_edge`` slightly (like ±0.001) or check out the full migration guide
    for a function to vendor.

    To approximate the old (``skimage``, v1.x) behavior after switching to
    ``skimage2``:

    * Instead of ``radius``, use ``radii=(radius,) * 3``
    * Instead of ``strict_radius=True`` (default), use ``adjust_edge=0.001``
    * Instead of ``strict_radius=False``, use ``adjust_edge=0.5``

    If you used ``decomposition='sequence'``, instead use
    `skimage2.morphology.footprint_disk_decomposed` to generate the decomposed
    footprint.

    <!--- cond-start: doc -->
    For example:

    >>> import numpy as np
    >>> import skimage as ski
    >>> import _skimage2 as ski2
    >>> radius = 7

    >>> fp1_strict = ski.morphology.ball(radius)
    >>> fp2_strict = ski2.morphology.footprint_ellipse((radius,) * 3, adjust_edge=0.001)
    >>> np.testing.assert_equal(fp1_strict, fp2_strict)

    >>> fp1 = ski.morphology.ball(radius, strict_radius=False)
    >>> fp2 = ski2.morphology.footprint_ellipse((radius,) * 3, adjust_edge=0.5)
    >>> np.testing.assert_equal(fp1, fp2)

    >>> fp1_sequence = ski.morphology.ball(radius, decomposition="sequence")
    >>> fp2_sequence = ski2.morphology.footprint_disk_decomposed(radius, ndim=3)
    >>> np.testing.assert_equal(fp1_sequence, fp2_sequence)

    .. admonition:: Reproduce exact results of old implementation
        :class: note dropdown

        Use/vendor the following function:

        .. code:: python

            def ball(radius, strict_radius=True):
                n = 2 * radius + 1
                Z, Y, X = np.mgrid[
                    -radius : radius : n * 1j,
                    -radius : radius : n * 1j,
                    -radius : radius : n * 1j,
                ]
                s = X**2 + Y**2 + Z**2
                if not strict_radius:
                    radius += 0.5
                return np.array(s <= radius**2, dtype=dtype)

    <!--- cond-end -->

    Other keyword parameters can be left unchanged.
    """,
    qname_old="skimage.morphology.ball",
    qname_new="skimage.morphology.footpint_ellipse",
)
def ball(radius, dtype=np.uint8, *, strict_radius=True, decomposition=None):
    """Generates a ball-shaped footprint.

    This is the 3D equivalent of a disk.
    A pixel is within the neighborhood if the Euclidean distance between
    it and the origin is no greater than radius.

    Parameters
    ----------
    radius : float
        The radius of the ball-shaped footprint.

    Other Parameters
    ----------------
    dtype : dtype-like, optional
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
        # Intentionally left this old implementation in place so that
        # compatibility tests comparing skimage and skimage2 behavior are
        # meaningful
        n = 2 * radius + 1
        Z, Y, X = np.mgrid[
            -radius : radius : n * 1j,
            -radius : radius : n * 1j,
            -radius : radius : n * 1j,
        ]
        s = X**2 + Y**2 + Z**2
        if not strict_radius:
            radius += 0.5
        return np.array(s <= radius**2, dtype=dtype)
    elif decomposition == 'sequence':
        sequence = _ski2_footprint_disk_decomposed(radius, ndim=3, dtype=dtype)
    else:
        raise ValueError(f"Unrecognized decomposition: {decomposition}")
    return sequence
