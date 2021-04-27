"""
Module defining a number of functions to quantify the overlap between shapes.

For instance BoundingBoxes used for object-detection.
"""
from __future__ import annotations

import numpy as np


class BoundingBox:
    """
    Construct an axis-aligned Bounding-Box
    consisting of top left and bottom right corners.

    The contructor uses the (r,c,..) coordinates for the top left corner and
    either the coordinates of the botton right corner or the BoundingBox
    dimensions (height, width,...).

    BoundingBoxes can have 2 (rectangle), 3 (3D BoundingBox) or more dimensions.

    Parameters
    ----------
    top_left : array-like of ints or floats
        (r,c)-coordinates for the top left corner of the BoundingBox.

    bottom_right : array-like of ints or floats, optional
        (r,c)-coordinates for the bottom right corner of the BoundingBox.

    dimensions : array-like of ints or floats, optional
        dimensions of the BoundingBox (height, width). The default is None.

    Raises
    ------
    ValueError
        If neither or both of bottom_right and dimensions are provided.

    Attributes
    ----------
    top_left : array of int or float
        The top left corner of the BoundingBox.
    bottom_right : array of int or float
        The bottom right corner of the BoundingBox.
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
        # use negative indexing in anticipation of nD hyperBoundingBoxes.
        return self.bottom_right[-2] - self.top_left[-2]

    @property
    def width(self):
        # use negative indexing in anticipation of nD hyperBoundingBoxes.
        return self.bottom_right[-1] - self.top_left[-1]

    @property
    def ndim(self):
        return len(self.top_left)

    def __bool__(self):
        return True

    def __eq__(self, other: BoundingBox):
        """Return true if 2 BoundingBoxes have the same position and dimension."""
        if not isinstance(other, BoundingBox):
            raise TypeError(
                    'Equality can only be checked with another BoundingBox'
                    )

        return (np.all(self.top_left == other.top_left)
                and np.all(self.bottom_right == other.bottom_right))

    def __repr__(self):
        return (f'BoundingBox({tuple(self.top_left)}, '
                f'bottom_right={tuple(self.bottom_right)})')

    def __str__(self):
        return self.__repr__()

    @property
    def area(self):
        """Return the area of a 2D BoundingBox."""
        if self.ndim == 2:
            return self.integral

        raise NotImplementedError("Area is only defined for 2D.")

    @property
    def volume(self):
        """Return the volume of a 3D BoundingBox."""
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
        """Return the dimensions of the BoundingBox as an array.

        Examples
        --------
        >>> r = BoundingBox((1, 1), bottom_right=(2, 3))
        >>> r.dimensions
        array([1, 2])
        """
        return self.bottom_right - self.top_left


def disjoint(bbox1, bbox2):
    """Check whether two BoundingBoxes are disjoint.

    Adapted from post from Aman Gupta [1]_.

    Parameters
    ----------
    bbox1, bbox2 : BoundingBox
        Input BoundingBoxes.

    Returns
    -------
    disjoint : bool
        True if the BoundingBoxes don't share any points.

    References
    ----------
    .. [1] https://www.geeksforgeeks.org/find-two-rectangles-overlap/
    """
    disjoint = (
            np.any(bbox1.bottom_right < bbox2.top_left)
            or np.any(bbox2.bottom_right < bbox1.top_left)
            )
    return disjoint


def intersect(bbox1, bbox2):
    """
    Return True if the 2 BoundingBoxes share at least one point.

    intersect is thus True whenever the BoundingBoxes are not disjoint.
    However intersecting BoundingBoxes may not overlap (ex: when only a corner or border is common).
    """
    return not disjoint(bbox1, bbox2)


def overlap(bbox1, bbox2):
    """
    Return the BoundingBox corresponding to the region of overlap between 2 BoundingBoxes,
    when the intersection of BoundingBoxes is more than 1-dimensional.
    BoundingBoxes with one corner or one side in common thus intersect but do not overlap.
    However overlapping BoundingBoxes always intersect.

    Return None if the two BoundingBoxes do not overlap.

    Parameters
    ----------
    bbox1, bbox2 : BoundingBox
        Input BoundingBoxes.

    Returns
    -------
    overlap : BoundingBox
        The BoundingBox produced by the overlap of ``bbox1`` and
        ``bbox2`` or None if the BoundingBoxes are not overlapping.

    Examples
    --------
    >>> r0 = BoundingBox((0, 0), bottom_right=(2, 3))
    >>> r1 = BoundingBox((1, 2), bottom_right=(4, 4))
    >>> intersect(r0, r1)
    BoundingBox((1, 2), bottom_right=(2, 3))

    >>> r2 = BoundingBox((10, 10), dimensions=(3, 3))
    >>> if overlap(r1, r2) is None:
    ...     print('r1 and r2 are not overlapping')
    r1 and r2 are not overlapping
    """
    if (np.any(bbox1.bottom_right <= bbox2.top_left) or
        np.any(bbox2.bottom_right <= bbox1.top_left)):
        return None  # below or equal contrary to disjoint: strictly below

    new_top_left = np.maximum(bbox1.top_left, bbox2.top_left)
    new_bottom_right = np.minimum(
            bbox1.bottom_right, bbox2.bottom_right
            )
    return BoundingBox(new_top_left, bottom_right=new_bottom_right)


def intersection_over_union(bbox1, bbox2):
    """
    Ratio intersection over union for a pair of BoundingBoxes.

    The intersection over union (IoU) ranges between 0 (no overlap) and 1 (full
    overlap) and has no unit.
    For 2D BoundingBoxes, the IoU corresponds to a ratio of areas.
    For 3D BoundingBoxes, the IoU corresponds to a ratio of volumes.
    For higher dimensions, the IoU corresponds to a ratio of the BoundingBox integrals.

    Parameters
    ----------
    bbox1, bbox2: BoundingBox
        The input BoundingBoxes.

    Returns
    -------
    iou : float
        The intersection over union value.
    """
    overlap_bbox = overlap(bbox1, bbox2)
    if overlap_bbox is None:
        return 0
    union_integral = (bbox1.integral +
                      bbox2.integral -
                      overlap_bbox.integral)

    return overlap_bbox.integral / union_integral
