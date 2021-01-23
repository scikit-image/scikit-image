"""
Module defining a number of functions to quantify the overlap between shapes, e.g. rectangles representing detections by bounding-boxes.

"""
def rectangle_area(topLeft, bottomRight):
    """Compute the area of a rectangle."""
    return (bottomRight[0] - topLeft[0] + 1) * (bottomRight[1] - topLeft[1] + 1)

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
    topLeft1, bottomRight1 = rectangle1
    topLeft2, bottomRight2 = rectangle2
    
    # If one rectangle is on left side of other 
    if topLeft1[1] >= bottomRight2[1] or topLeft2[1] >= bottomRight1[1]: 
        return False
      
    # If one rectangle is above other 
    if topLeft1[0] >= bottomRight2[0] or topLeft2[0] >= bottomRight1[0]: 
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
   
    topLeft1, bottomRight1 = rectangle1
    topLeft2, bottomRight2 = rectangle2
    
    # determine the (x, y)-coordinates of the top left and bottom right points of the intersection rectangle
    topLeft_r = max(topLeft1[0], topLeft2[0])
    topLeft_c = max(topLeft1[1], topLeft2[1])
    bottomRight_r = min(bottomRight1[0], bottomRight2[0])
    bottomRight_c = min(bottomRight1[1], bottomRight2[1])
    
    return (topLeft_r, topLeft_c), (bottomRight_r, bottomRight_c)

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
    """Compute the area for the rectangle corresponding to the union of 2 rectangles."""
    return (rectangle_area(rectangle1) 
            + rectangle_area(rectangle2) 
            - intersection_area_rectangles(rectangle1 , rectangle2) )

def intersection_over_union_rectangles(rectangle1, rectangle2):
    """
    Compute the ratio intersection aera over union area for a pair of rectangle.
    
    The intersection over union (IoU) ranges between 0 (no overlap) and 1 (full overlap).
    """
    return intersection_area_rectangles(rectangle1, rectangle2) / union_area_rectangles(rectangle1, rectangle2)


if __name__ == "__main__":
    rectangle1 = ((0,0),(2,4))
    rectangle2 = ((1,3),(3,6))
    
    assert rectangle_area(*rectangle1) == 3*5
    assert rectangle_area(*rectangle2) == 3*4
    
    assert intersection_rectangles(rectangle1, rectangle2) == ((1,3), (2,4))
