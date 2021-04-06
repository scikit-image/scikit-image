"""
Module defining a number of functions to quantify the overlap between shapes.
for instance rectangles representing detections by bounding-boxes.

"""
from __future__ import annotations

import numpy as np


class Rectangle:
    """
    Construct a rectangle using the (r,c) coordinates for the top left corner,
    and either the coordinates of the botton right corner
    or the rectangle dimensions (height, width).

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
        If none or both of bottom_right and dimensions are provided.

    Returns
    -------
    Rectangle object.
    """

    def __init__(self, top_left, *, bottom_right=None, dimensions=None):
        self.top_left = np.asarray(top_left)

        if (bottom_right is None) and (dimensions is None):
            raise ValueError("Specify one of bottom_right or dimensions.")

        if (bottom_right is not None) and (dimensions is not None):
            raise ValueError("Specify bottom_right or dimensions, not both.")

        if bottom_right is not None:
            self.bottom_right = np.asarray(bottom_right)

        elif dimensions is not None:
            self.bottom_right = self.top_left + np.asarray(dimensions)

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

    @property
    def area(self):
        """Return the rectangle area in pixels."""
        return np.prod(self.dimensions)

    @property
    def dimensions(self):
        """Return the (height, width) dimensions in pixels."""
        return self.bottom_right - self.top_left


def _disjoint(rectangle1, rectangle2):
    """Check whether two rectangles are disjoint

    Adapted from post from Aman Gupta [1]_.

    Parameters
    ----------
    rectangle1, rectangle2 : Rectangle
        Input rectangles.

    Returns
    -------
    disjoint : bool
        True if the rectangles are don't share any points.

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
    """Return a Rectangle corresponding to the intersection between 2 rectangles.

    Parameters
    ----------
    rectangle1, rectangle2 : Rectangle
        Input rectangles.

    Returns
    -------
    intersected : Rectangle
        The rectangle produced by intersecting ``rectangle1`` and
        ``rectangle2``.

    Raises
    ------
    ValueError
        If the rectangles are not intersecting.
        Use is_intersecting to first test if the rectangles are intersecting.
    """
    if _disjoint(rectangle1, rectangle2):
        # return the "null rectangle" if they are disjoint
        ndim = rectangle1.ndim
        return Rectangle((0,) * ndim, bottom_right=(0,) * ndim)
    new_top_left = np.maximum(rectangle1.top_left, rectangle2.top_left)
    new_bottom_right = np.minimum(
            rectangle1.bottom_right, rectangle2.bottom_right
            )
    return Rectangle(new_top_left, bottom_right=new_bottom_right)


def intersection_over_union(rectangle1, rectangle2):
    """Ratio intersection area over union area for a pair of rectangles.

    The intersection over union (IoU) ranges between 0 (no overlap) and 1 (full
    overlap).

    Parameters
    ----------
    rectangle1, rectangle2: Rectangle
        The input rectangles.

    Returns
    -------
    iou : float
        The intersection over union value.
    """
    intersection_area = intersect(rectangle1, rectangle2).area
    union_area = rectangle1.area + rectangle2.area - intersection_area
    return intersection_area / union_area
