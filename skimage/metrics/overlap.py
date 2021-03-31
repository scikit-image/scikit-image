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
    top_left : tuple of 2 ints
        (r,c)-coordinates for the top left corner of the rectangle.

    bottom_right : tuple of 2 ints, optional, default=None
        (r,c)-coordinates for the bottom right corner of the rectangle.

    dimensions : tuple of 2 ints, optional, default=None
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
            self.bottom_right = np.asarray(
                    [top + size - 1 for top, size in zip(top_left, dimensions)]
                    )

    @property
    def height(self):
        # use negative indexing in anticipation of nD hyperrectangles.
        return self.bottom_right[-2] - self.top_left[-2] + 1

    @property
    def width(self):
        # use negative indexing in anticipation of nD hyperrectangles.
        return self.bottom_right[-1] - self.top_left[-1] + 1

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
        return self.bottom_right - self.top_left + 1


def is_intersecting(rectangle1, rectangle2):
    """Check whether two rectangles intersect.

    Adapted from post from Aman Gupta [1]_.

    Parameters
    ----------
    rectangle1, rectangle2 : Rectangle
        Input rectangles.

    Returns
    -------
    intersecting : bool
        True if the rectangles are intersecting.

    References
    ----------
    .. [1] https://www.geeksforgeeks.org/find-two-rectangles-overlap/
    """
    disjoint = (
            np.any(rectangle1.bottom_right <= rectangle2.top_left)
            or np.any(rectangle2.bottom_right <= rectangle1.top_left)
            )
    return not disjoint


def intersection_rectangle(rectangle1, rectangle2):
    """
    Return a Rectangle corresponding to the intersection between 2 rectangles.

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
    if not is_intersecting(rectangle1, rectangle2):
        raise ValueError(
                'The rectangles are not intersecting. Use is_intersecting '
                'to first test if the rectangles are intersecting.'
                )

    new_top_left = np.maximum(rectangle1.top_left, rectangle2.top_left)
    new_bottom_right = np.minimum(
            rectangle1.bottom_right, rectangle2.bottom_right
            )
    return Rectangle(new_top_left, bottom_right=new_bottom_right)


def intersection_area(rectangle1, rectangle2):
    """Compute the intersection area between 2 rectangles.

    Parameters
    ----------
    rectangle1, rectangle2 : Rectangle
        Input rectangles.

    Returns
    -------
    intersection : float
        The intersection area.
    """
    if not is_intersecting(rectangle1, rectangle2):
        return 0

    # Compute area of the intersecting box
    return intersection_rectangle(rectangle1, rectangle2).area


def union_area(rectangle1, rectangle2):
    """Compute the area corresponding to the union of 2 rectangles.

    Parameters
    ----------
    rectangle1, rectangle2 : Rectangle
        Input rectangles.

    Returns
    -------
    union : float
        The union area.
    """
    return (rectangle1.area + rectangle2.area
            - intersection_area(rectangle1, rectangle2))


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
    return (intersection_area(rectangle1, rectangle2)
            / union_area(rectangle1, rectangle2))
