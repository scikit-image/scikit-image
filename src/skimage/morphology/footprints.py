import numpy as np

from _skimage2.morphology.footprints import (
    ball as ball,
    diamond as diamond,
    disk as disk,
    ellipse as ellipse,
    footprint_from_sequence as footprint_from_sequence,
    footprint_rectangle as _sk2_footprint_rectangle,
    footprint_rectangle_decomposed as _sk2_footprint_rectangle_decompose,
    octagon as octagon,
    octahedron as octahedron,
    star as star,
)  # noqa: F401
from _skimage2.morphology._footprints import mirror_footprint, pad_footprint  # noqa: F401

from .._migration import ski2_migration_decorator

from skimage._doctest_adapters import adapt_doctests

adapt_doctests(globals())


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
        footprint = _sk2_footprint_rectangle(shape=shape, dtype=dtype)
    else:
        footprint = _sk2_footprint_rectangle_decompose(
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
