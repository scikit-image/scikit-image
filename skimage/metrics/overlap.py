"""
Module defining a number of functions to quantify the overlap between shapes.
for instance rectangles representing detections by bounding-boxes.

"""

class Rectangle():
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
        self._top_left = top_left
        self._r = top_left[0]
        self._c = top_left[1]

        if (bottom_right is None) and (dimensions is None):
            raise ValueError("One of bottom_right or dimensions argument should be provided.")

        if (bottom_right is not None) and (dimensions is not None):
            raise ValueError("Either specify the bottom_right or dimensions, not both.")

        if bottom_right is not None:
            self._bottom_right = bottom_right
            self._height = self._bottom_right[0] - self._top_left[0] + 1
            self._width  = self._bottom_right[1] - self._top_left[1] + 1

        elif dimensions is not None:
            self._height, self._width = dimensions
            self._bottom_right = (self._r + self._height - 1,
                                 self._c + self._width - 1)

    def __eq__(self, rectangle2):
        """Return true if 2 rectangles have the same position and dimension."""
        if not isinstance(rectangle2, Rectangle):
            return False

        if (self._top_left == rectangle2._top_left and
           self._bottom_right == rectangle2._bottom_right):
            return True

        return False

    def get_top_left(self):
        """Returns the (r,c)-coordinates for the top left corner."""
        return self._top_left

    def get_bottom_right(self):
        """Returns the (r,c)-coordinates for the bottom right corner."""
        return self._bottom_right

    def get_area(self):
        """Return the rectangle area in pixels."""
        return self._height * self._width

    def get_dimensions(self):
        """Return the (height, width) dimensions in pixels."""
        return self._height, self._width


    @staticmethod
    def is_intersecting(rectangle1, rectangle2):
        """
        Check if 2 rectangles are intersecting.

        Adapted from post from Aman Gupta
        at https://www.geeksforgeeks.org/find-two-rectangles-overlap/.

        Parameters
        ----------
        rectangle1 : a Rectangle object
            First rectangle.
        rectangle2 : a second Rectangle object
            Second rectangle

        Returns
        -------
        True if the rectangles are intersecting.
        """
        # If one rectangle is on left side of other
        if (rectangle1._top_left[1] >= rectangle2._bottom_right[1] or
           rectangle2._top_left[1] >= rectangle1._bottom_right[1]):
            return False

        # If one rectangle is above other
        if (rectangle1._top_left[0] >= rectangle2._bottom_right[0] or
           rectangle2._top_left[0] >= rectangle1._bottom_right[0]):
            return False

        return True

    @staticmethod
    def intersection_rectangles(rectangle1, rectangle2):
        """
        Return a Rectangle corresponding to the intersection between 2 rectangles.

        Parameters
        ----------
        rectangle1 : a Rectangle object
            First rectangle.
        rectangle2 : a second Rectangle object
            Second rectangle

        Raises
        ------
        ValueError
            If the rectangles are not intersecting.
            Use is_intersecting to first test if the rectangles are intersecting.

        Returns
        -------
        Rectangle object representing the intersection.
        """
        if not Rectangle.is_intersecting(rectangle1, rectangle2):
            raise ValueError("""The rectangles are not intersecting.
                             Use is_intersecting to first test if the rectangles are intersecting.""")

        # determine the (r, c)-coordinates of the top left and bottom right corners
        # for the intersection rectangle
        r = max(rectangle1._r, rectangle2._r)
        c = max(rectangle1._c, rectangle2._c)
        bottom_right_r = min(rectangle1._bottom_right[0], rectangle2._bottom_right[0])
        bottom_right_c = min(rectangle1._bottom_right[1], rectangle2._bottom_right[1])

        return Rectangle((r, c), bottom_right=(bottom_right_r, bottom_right_c))

    @staticmethod
    def intersection_area(rectangle1, rectangle2):
        """
        Return the intersection area between 2 rectangles.

        Parameters
        ----------
        rectangle1 : Rectangle
        rectangle2 : Rectangle

        Returns
        -------
        Intersection, float
            a float value corresponding to the intersection area.
        """
        if not Rectangle.is_intersecting(rectangle1, rectangle2):
            return 0

        # Compute area of the intersecting box
        return Rectangle.intersection_rectangles(rectangle1, rectangle2).get_area()

    @staticmethod
    def union_area(rectangle1, rectangle2):
        """Compute the area for the rectangle corresponding to the union of 2 rectangles."""
        return (rectangle1.get_area()
                + rectangle2.get_area()
                - Rectangle.intersection_area(rectangle1, rectangle2))

    @staticmethod
    def intersection_over_union(rectangle1, rectangle2):
        """
        Compute the ratio intersection aera over union area for a pair of rectangle.

        The intersection over union (IoU) ranges between 0 (no overlap) and 1 (full overlap).
        """
        return Rectangle.intersection_area(rectangle1, rectangle2) / Rectangle.union_area(rectangle1, rectangle2)


if __name__ == "__main__":

    height1, width1 = 2, 4
    rectangle1 = Rectangle((0, 0), dimensions=(height1, width1))  # bottom right in (1,3) thus

    height2 = width2 = 3
    rectangle2 = Rectangle((1, 3), dimensions=(height2, width2))

    rectangle3 = Rectangle((0, 0), bottom_right=(2, 2))

    rectangle4 = Rectangle((10, 10), dimensions=(5, 5))


    # Check area
    assert rectangle1.get_area() == height1 * width1
    assert rectangle2.get_area() == height2 * width2
    assert rectangle3.get_area() == 3*3

    # Check other dimension (the one not set in constructor)
    assert rectangle1.get_bottom_right() == (1, 3)
    assert rectangle3.get_dimensions() == (3, 3)

    # Intersections
    assert not Rectangle.is_intersecting(rectangle1, rectangle2)
    assert Rectangle.is_intersecting(rectangle1, rectangle3)

    # Intersection rectangle and == comparison
    rect_inter13 = Rectangle.intersection_rectangles(rectangle1, rectangle3)
    assert  rect_inter13 == Rectangle((0, 0), bottom_right=(1, 2))
    assert rect_inter13 != 5
    #intersection_rectangles(rectangle1, rectangle2) # should raise an issue

    # Union area
    union_area13 = Rectangle.union_area(rectangle1, rectangle3)
    assert union_area13 == (rectangle1.get_area() + rectangle3.get_area() - rect_inter13.get_area())

    # IoU
    assert Rectangle.intersection_over_union(rectangle1, rectangle3) == rect_inter13.get_area()/union_area13
