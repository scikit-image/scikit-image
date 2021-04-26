"""
Module defining a number of functions to quantify the overlap between shapes.
for instance rectangles representing detections by bounding-boxes.

"""
from __future__ import annotations

import numpy as np


class Rectangle:
    """Construct a rectangle consisting of top left and bottom right corners.

    The contructor uses the (r,c) coordinates for the top left corner and
    either the coordinates of the botton right corner or the rectangle
    dimensions (height, width).

    Parameters
    ----------
    top_left : array-like of ints or floats
        (r,c)-coordinates for the top left corner of the rectangle.

    bottom_right : array-like of ints or floats, optional
        (r,c)-coordinates for the bottom right corner of the rectangle.

    dimensions : array-like of ints or floats, optional
        dimensions of the rectangle (height, width). The default is None.

    Raises
    ------
    ValueError
        If neither or both of bottom_right and dimensions are provided.

    Attributes
    ----------
    top_left : array of int or float
        The top left corner of the rectangle.
    bottom_right : array of int or float
        The bottom right corner of the rectangle.

    Notes
    -----
    ``bool(rectangle)`` will be False when the rectangle has area 0.
    """

    def __init__(self, top_left, *, bottom_right=None, dimensions=None):
        self.top_left = np.asarray(top_left)

        if (bottom_right is None) and (dimensions is None):
            raise ValueError("Specify one of bottom_right or dimensions.")

        if (bottom_right is not None) and (dimensions is not None):
            raise ValueError("Specify bottom_right or dimensions, not both.")

        if bottom_right is not None:
            if not np.all(top_left <= bottom_right):
                raise ValueError("Bottom right corner should have coordinates "
                                 "larger or equal to the top left corner.")
            self.bottom_right = np.asarray(bottom_right)

        elif dimensions is not None:
            dimensions = np.asarray(dimensions)
            if not (dimensions >= 0).all():
                raise ValueError("Dimensions should be positive.")
            self.bottom_right = self.top_left + dimensions

    @property
    def height(self):
        # use negative indexing in anticipation of nD hyperrectangles.
        return self.bottom_right[-2] - self.top_left[-2]

    @property
    def width(self):
        # use negative indexing in anticipation of nD hyperrectangles.
        return self.bottom_right[-1] - self.top_left[-1]

    @property
    def ndim(self):
        return len(self.top_left)

    def __bool__(self):
        return bool(self.area > 0)  # cast needed to avoid np.bool_

    def __eq__(self, other: Rectangle):
        """Return true if 2 rectangles have the same position and dimension."""
        if not isinstance(other, Rectangle):
            raise TypeError(
                    'Equality can only be checked with another Rectangle'
                    )

        return (np.all(self.top_left == other.top_left)
                and np.all(self.bottom_right == other.bottom_right))

    def __repr__(self):
        return (f'Rectangle({tuple(self.top_left)}, '
                f'bottom_right={tuple(self.bottom_right)})')

    def __str__(self):
        return self.__repr__()

    @property
    def area(self):
        """Return the area of a 2D rectangle."""
        if self.ndim == 2:
            return self.integral

        raise NotImplementedError("Area is only defined for 2D.")

    @property
    def volume(self):
        """Return the volume of a 3D rectangle."""
        if self.ndim == 3:
            return self.integral

        raise NotImplementedError("Volume is only defined for 3D.")

    @property
    def integral(self):
        """
        Return the integral of the shape along all dimensions of the shape.

        For 2D/3D shapes, the integral corresponds respectively to the area/volume.
        """
        return np.prod(self.dimensions)

    @property
    def dimensions(self):
        """Return the dimensions of the rectangle as an array.

        Examples
        --------
        >>> r = Rectangle((1, 1), bottom_right=(2, 3))
        >>> r.dimensions
        array([1, 2])
        """
        return self.bottom_right - self.top_left


def _disjoint(rectangle1, rectangle2):
    """Check whether two rectangles are disjoint.

    Adapted from post from Aman Gupta [1]_.

    Parameters
    ----------
    rectangle1, rectangle2 : Rectangle
        Input rectangles.

    Returns
    -------
    disjoint : bool
        True if the rectangles don't share any points.

    References
    ----------
    .. [1] https://www.geeksforgeeks.org/find-two-rectangles-overlap/
    """
    disjoint = (
            np.any(rectangle1.bottom_right < rectangle2.top_left)
            or np.any(rectangle2.bottom_right < rectangle1.top_left)
            )
    return disjoint


def intersect(rectangle1, rectangle2):
    """
    Return True if the 2 rectangles share at least one point.

    intersect is thus True whenever the rectangles are not disjoint.
    However intersecting rectangles may not overlap (ex: when only a corner or border is common).
    """
    return not _disjoint(rectangle1, rectangle2)


def overlap(rectangle1, rectangle2):
    """
    Return the Rectangle corresponding to the region of overlap between 2 rectangles,
    when the intersection of rectangles is more than 1-dimensional.
    Rectangles with one corner or one side in common thus intersect but do not overlap.
    However overlapping rectangles always intersect.

    Return None if the two rectangles do not overlap.

    Parameters
    ----------
    rectangle1, rectangle2 : Rectangle
        Input rectangles.

    Returns
    -------
    overlap : Rectangle
        The rectangle produced by the overlap of ``rectangle1`` and
        ``rectangle2`` or None if the rectangles are not overlapping.

    Examples
    --------
    >>> r0 = Rectangle((0, 0), bottom_right=(2, 3))
    >>> r1 = Rectangle((1, 2), bottom_right=(4, 4))
    >>> intersect(r0, r1)
    Rectangle((1, 2), bottom_right=(2, 3))

    >>> r2 = Rectangle((10, 10), dimensions=(3, 3))
    >>> if overlap(r1, r2) is None:
    ...     print('r1 and r2 are not overlapping')
    r1 and r2 are not overlapping
    """
    if (np.any(rectangle1.bottom_right <= rectangle2.top_left) or
        np.any(rectangle2.bottom_right <= rectangle1.top_left)):
        return None  # below or equal contrary to disjoint: strictly below

    new_top_left = np.maximum(rectangle1.top_left, rectangle2.top_left)
    new_bottom_right = np.minimum(
            rectangle1.bottom_right, rectangle2.bottom_right
            )
    return Rectangle(new_top_left, bottom_right=new_bottom_right)


def intersection_over_union(rectangle1, rectangle2):
    """
    Ratio intersection over union for a pair of rectangles.

    The intersection over union (IoU) ranges between 0 (no overlap) and 1 (full
    overlap) and has no unit.
    For 2D rectangles, the IoU corresponds to a ratio of areas.
    For 3D rectangles, the IoU corresponds to a ratio of volumes.
    For higher dimensions, the IoU corresponds to a ratio of the shape integrals.

    Parameters
    ----------
    rectangle1, rectangle2: Rectangle
        The input rectangles.

    Returns
    -------
    iou : float
        The intersection over union value.
    """
    overlap_rectangle = overlap(rectangle1, rectangle2)
    if overlap_rectangle is None:
        return 0
    union_integral = (rectangle1.integral +
                      rectangle2.integral -
                      overlap_rectangle.integral)

    return overlap_rectangle.integral / union_integral
