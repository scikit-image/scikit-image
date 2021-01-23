"""
This module defines a number of functions to quantify the overlap between shapes, e.g. rectangles representing detections by bounding-boxes.

"""
def rectangle_area(topLeft_r, topLeft_c, bottomRight_r, bottomRight_c):
    """Compute the area of a rectangle."""
    return (bottomRight_c - topLeft_c + 1) * (bottomRight_r - topLeft_r + 1)

def isRectangleIntersecting(rectangle1, rectangle2):
    """
    Check if 2 rectangles are intersecting.
    
    The rectangle should be encoded as the (topLeft_r, topLeft_c, bottomRight_r, bottomRight_c) for the coordinates of the top left (l) and bottom right (r) corners of the rectangle.
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
    topLeft_r1,  topLeft_c1, bottomRight_r1, bottomRight_c1 = rectangle1
    topLeft_r2,  topLeft_c2, bottomRight_r2, bottomRight_c2 = rectangle2
    
    # If one rectangle is on left side of other 
    if topLeft_c1 >= bottomRight_c2 or topLeft_c2 >= bottomRight_c1: 
        return False
      
    # If one rectangle is above other 
    if topLeft_r1 >= bottomRight_r2 or topLeft_r2 >= bottomRight_r1: 
        return False
    
    return True

def intersection_rectangles(rectangle1, rectangle2):
    """
    Compute the coordinates of a rectangle at the intersection between 2 rectangles.
    
    The rectangle should be encoded as the (topLeft_r, topLeft_c, bottomRight_r, bottomRight_c) for the coordinates of the top left (l) and bottom right (r) corners of the rectangle.

    Parameters
    ----------
    rectangle1 : tuple of 4 integers
    
    rectangle2 : TYPE
        DESCRIPTION.

    Returns
    -------
    (topLeft_c,topLeft_r,bottomRight_c,bottomRight_r) tuple for the  coordinates of the intersecting rectangle.

    """    
    if not isRectangleIntersecting(rectangle1, rectangle2): raise ValueError("The rectangles are not intersecting")
   
    topLeft_r1,  topLeft_c1, bottomRight_r1, bottomRight_c1 = rectangle1
    topLeft_r2,  topLeft_c2, bottomRight_r2, bottomRight_c2 = rectangle2
    
    # determine the (x, y)-coordinates of the top left and bottom right points of the intersection rectangle
    topLeft_r = max(topLeft_r1, topLeft_r2)
    topLeft_c = max(topLeft_c1, topLeft_c2)
    bottomRight_r = min(bottomRight_r1, bottomRight_r2)
    bottomRight_c = min(bottomRight_c1, bottomRight_c2)
    
    return topLeft_r, topLeft_c, bottomRight_r, bottomRight_c

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