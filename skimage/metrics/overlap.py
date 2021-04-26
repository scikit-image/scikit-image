"""
Module defining a number of functions to quantify the overlap between shapes.
for instance BBoxes representing detections by bounding-boxes.

"""
from __future__ import annotations

import numpy as np


class BBox:
    """
    Construct an axis-aligned Bounding-Box
    consisting of top left and bottom right corners.

    The contructor uses the (r,c,..) coordinates for the top left corner and
    either the coordinates of the botton right corner or the BBox
    dimensions (height, width,...).

    BBoxes can have 2 (rectangle), 3 (3D BBox) or more dimensions.

    Parameters
    ----------
    top_left : array-like of ints or floats
        (r,c)-coordinates for the top left corner of the BBox.

    bottom_right : array-like of ints or floats, optional
        (r,c)-coordinates for the bottom right corner of the BBox.

    dimensions : array-like of ints or floats, optional
        dimensions of the BBox (height, width). The default is None.

    Raises
    ------
    ValueError
        If neither or both of bottom_right and dimensions are provided.

    Attributes
    ----------
    top_left : array of int or float
        The top left corner of the BBox.
    bottom_right : array of int or float
        The bottom right corner of the BBox.

    Notes
    -----
    ``bool(BBox)`` will be False when the BBox has area 0.
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
        # use negative indexing in anticipation of nD hyperBBoxes.
        return self.bottom_right[-2] - self.top_left[-2]

    @property
    def width(self):
        # use negative indexing in anticipation of nD hyperBBoxes.
        return self.bottom_right[-1] - self.top_left[-1]

    @property
    def ndim(self):
        return len(self.top_left)

    def __bool__(self):
        return bool(self.area > 0)  # cast needed to avoid np.bool_

    def __eq__(self, other: BBox):
        """Return true if 2 BBoxes have the same position and dimension."""
        if not isinstance(other, BBox):
            raise TypeError(
                    'Equality can only be checked with another BBox'
                    )

        return (np.all(self.top_left == other.top_left)
                and np.all(self.bottom_right == other.bottom_right))

    def __repr__(self):
        return (f'BBox({tuple(self.top_left)}, '
                f'bottom_right={tuple(self.bottom_right)})')

    def __str__(self):
        return self.__repr__()

    @property
    def area(self):
        """Return the area of a 2D BBox."""
        if self.ndim == 2:
            return self.integral

        raise NotImplementedError("Area is only defined for 2D.")

    @property
    def volume(self):
        """Return the volume of a 3D BBox."""
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
        """Return the dimensions of the BBox as an array.

        Examples
        --------
        >>> r = BBox((1, 1), bottom_right=(2, 3))
        >>> r.dimensions
        array([1, 2])
        """
        return self.bottom_right - self.top_left


def _disjoint(BBox1, BBox2):
    """Check whether two BBoxs are disjoint.

    Adapted from post from Aman Gupta [1]_.

    Parameters
    ----------
    BBox1, BBox2 : BBox
        Input BBoxes.

    Returns
    -------
    disjoint : bool
        True if the BBoxes don't share any points.

    References
    ----------
    .. [1] https://www.geeksforgeeks.org/find-two-rectangles-overlap/
    """
    disjoint = (
            np.any(BBox1.bottom_right < BBox2.top_left)
            or np.any(BBox2.bottom_right < BBox1.top_left)
            )
    return disjoint


def intersect(bbox1, bbox2):
    """
    Return True if the 2 BBoxes share at least one point.

    intersect is thus True whenever the BBoxes are not disjoint.
    However intersecting BBoxes may not overlap (ex: when only a corner or border is common).
    """
    return not _disjoint(bbox1, bbox2)


def overlap(bbox1, bbox2):
    """
    Return the BBox corresponding to the region of overlap between 2 BBoxes,
    when the intersection of BBoxes is more than 1-dimensional.
    BBoxes with one corner or one side in common thus intersect but do not overlap.
    However overlapping BBoxes always intersect.

    Return None if the two BBoxes do not overlap.

    Parameters
    ----------
    bbox1, bbox2 : BBox
        Input BBoxes.

    Returns
    -------
    overlap : BBox
        The BBox produced by the overlap of ``bbox1`` and
        ``bbox2`` or None if the BBoxes are not overlapping.

    Examples
    --------
    >>> r0 = BBox((0, 0), bottom_right=(2, 3))
    >>> r1 = BBox((1, 2), bottom_right=(4, 4))
    >>> intersect(r0, r1)
    BBox((1, 2), bottom_right=(2, 3))

    >>> r2 = BBox((10, 10), dimensions=(3, 3))
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
    return BBox(new_top_left, bottom_right=new_bottom_right)


def intersection_over_union(bbox1, bbox2):
    """
    Ratio intersection over union for a pair of BBoxes.

    The intersection over union (IoU) ranges between 0 (no overlap) and 1 (full
    overlap) and has no unit.
    For 2D BBoxes, the IoU corresponds to a ratio of areas.
    For 3D BBoxes, the IoU corresponds to a ratio of volumes.
    For higher dimensions, the IoU corresponds to a ratio of the BBox integrals.

    Parameters
    ----------
    bbox1, bbox2: BBox
        The input BBoxes.

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
