"""
This module defines a number of functions to quantify the overlap between shapes, e.g. rectangles representing detections by bounding-boxes.

"""
def rectangle_area(xl, yl, xr, yr):
    """Compute the area of a rectangle."""
    return (xr - xl + 1) * (yr - yl + 1)

def isRectangleIntersecting(rectangle1, rectangle2):
    """
    Check if 2 rectangles are intersecting.
    
    The rectangle should be encoded as the (xl, yl, xr, yr) for the coordinates of the top left (l) and bottom right (r) corners of the rectangle.
    Adapted from post from Aman Gupta at https://www.geeksforgeeks.org/find-two-rectangles-overlap/.
    
    Parameters
    ----------
    rectangle1 : tuple of 4 integers
        First rectangle.
    rectangle2 : tuple of 4 integers
        Second rectangle

    Returns
    -------
    True if the rectangle are intersecting.
    """
    xl1, yl1, xr1, yr1 = rectangle1
    xl2, yl2, xr2, yr2 = rectangle2
    
    # If one rectangle is on left side of other 
    if xl1 >= xr2 or xl2 >= xr1: 
        return False
      
    # If one rectangle is above other 
    if yl1 >= yr2 or yl2 >= yr1: 
        return False
    
    return True

def intersection_rectangles(rectangle1, rectangle2):
    """
    Compute the coordinates of a rectangle at the intersection between 2 rectangles.
    
    The rectangle should be encoded as the (xl, yl, xr, yr) for the coordinates of the top left (l) and bottom right (r) corners of the rectangle.

    Parameters
    ----------
    rectangle1 : tuple of 4 integers
    
    rectangle2 : TYPE
        DESCRIPTION.

    Returns
    -------
    (xl,yl,xr,yr) tuple for the  coordinates of the intersecting rectangle.

    """    
    if not isRectangleIntersecting(rectangle1, rectangle2): raise ValueError("The rectangles are not intersecting")
   
    xl1, yl1, xr1, yr1 = rectangle1
    xl2, yl2, xr2, yr2 = rectangle2
    
    # determine the (x, y)-coordinates of the top left and bottom right points of the intersection rectangle
    xl = max(xl1, xl2)
    yl = max(yl1, yl2)
    xr = min(xr1, xr2)
    yr = min(yr1, yr2)
    
    return xl, yl, xr, yr

def intersection_area_rectangles(rectangle1, rectangle2):
    """
    Return the intersection area between 2 rectangles.
    
    Parameters
    ----------
    rectangle1 : tuple of ints
        a first rectangle encoded as (x, y, width, height).
    rectangle2 : tuple of ints
         a second rectangle encoded as (x, y, width, height)..

    Returns
    -------
    Intersection, float
        a float value corresponding to the intersection area.
    """
    if not isRectangleIntersecting(rectangle1, rectangle2):
        return 0
    
    # Compute area of the intersecting box
    return rectangle_area( intersection_rectangles(rectangle1, rectangle2) )

def union_area_rectangles(rectangle1, rectangle2):
    """
    Compute the equivalent area for the union of 2 rectangles.
    """
    return (rectangle_area(rectangle1) 
            + rectangle_area(rectangle2) 
            - intersection_area_rectangles(rectangle1 , rectangle2) )

def intersection_over_union_rectangles(rectangle1, rectangle2):
    """
    Compute the ratio intersection aera over union area for a pair of rectangle.
    
    The intersection over union (IoU) ranges between 0 (no overlap) and 1 (full overlap).
    """
    return intersection_area_rectangles(rectangle1, rectangle2) / union_area_rectangles(rectangle1, rectangle2)