"""
Module defining a number of functions to quantify the overlap between shapes, e.g. rectangles representing detections by bounding-boxes.

"""

class Rectangle():
    """Represent a rectangular bounding-box."""
    
    def __init__(self, topLeft, bottomRight=None, size=None):
        """
        Construct a rectangle using the (r,c) coordinates for the top left corner, the coordinates of the botton right corner or the size (height, width).

        Parameters
        ----------
        topLeft : tuple
            (r,c)-coordinates for the top left corner of the rectangle.
        bottomRight : tuple, optional
            (r,c)-coordinates for the bottom right corner of the rectangle. The default is None.
        size : tuple, optional
            Size of the rectangle (height, width). The default is None.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        Rectangle object.

        """
        self.topLeft =topLeft
        self.r = topLeft[0]
        self.c = topLeft[1]
        
        if (bottomRight is None) and (size is None):
            raise ValueError("One of bottomRight or size argument should be defined.")
            
        elif (bottomRight is not None) and (size is not None):
            raise ValueError("Either specify the bottomRight or size.")
        
        elif bottomRight is not None:
            self.bottomRight = bottomRight
            self.height = self.bottomRight[0] - self.topLeft[0] +1
            self.width  = self.bottomRight[1] - self.topLeft[1] +1
            self.size = (self.height, self.width)
        
        elif size is not None:
            self.height, self.width = size
            self.size = size
            self.bottomRight = (self.r + self.height-1, 
                                self.c + self.width -1)
        
        self.area = self.height * self.width
        
def isRectangleIntersecting(rectangle1, rectangle2):
    """
    Check if 2 rectangles are intersecting.
    
    Adapted from post from Aman Gupta at https://www.geeksforgeeks.org/find-two-rectangles-overlap/.
    
    Parameters
    ----------
    rectangle1 : a Rectangle object
        First rectangle.
    rectangle2 : a second Rectangle object
        Second rectangle

    Returns
    -------
    True if the rectangle are intersecting.
    """
    # If one rectangle is on left side of other 
    if rectangle1.topLeft[1] >= rectangle2.bottomRight[1] or rectangle2.topLeft[1] >= rectangle1.bottomRight[1]: 
        return False
      
    # If one rectangle is above other 
    if rectangle1.topLeft[0] >= rectangle2.bottomRight[0] or rectangle2.topLeft[0] >= rectangle1.bottomRight[0]: 
        return False
    
    return True

def intersection_rectangles(rectangle1, rectangle2):
    """
    Return a rectangle object corresponding to the intersection between 2 rectangles.
    
    Parameters
    ----------
    rectangle1 : tuple of 4 integers
    
    rectangle2 : TYPE
        DESCRIPTION.

    Returns
    -------
    Rectangle object representing the intersection.
    Raise Value error if no interesection.
    """    
    if not isRectangleIntersecting(rectangle1, rectangle2): 
        raise ValueError("The rectangles are not intersecting")
       
    # determine the (x, y)-coordinates of the top left and bottom right points of the intersection rectangle
    r = max(rectangle1.r, rectangle2.r)
    c = max(rectangle1.c, rectangle2.c)
    bottomRight_r = min(rectangle1.bottomRight[0], rectangle2.bottomRight[0])
    bottomRight_c = min(rectangle1.bottomRight[1], rectangle2.bottomRight[1])
    
    return Rectangle((r,c), (bottomRight_r, bottomRight_c))

def intersection_area_rectangles(rectangle1, rectangle2):
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
    if not isRectangleIntersecting(rectangle1, rectangle2):
        return 0
    
    # Compute area of the intersecting box
    return intersection_rectangles(rectangle1, rectangle2).area

def union_area_rectangles(rectangle1, rectangle2):
    """Compute the area for the rectangle corresponding to the union of 2 rectangles."""
    return (  rectangle1.area
            + rectangle2.area 
            - intersection_area_rectangles(rectangle1 , rectangle2) )

def intersection_over_union_rectangles(rectangle1, rectangle2):
    """
    Compute the ratio intersection aera over union area for a pair of rectangle.
    
    The intersection over union (IoU) ranges between 0 (no overlap) and 1 (full overlap).
    """
    return intersection_area_rectangles(rectangle1, rectangle2) / union_area_rectangles(rectangle1, rectangle2)


if __name__ == "__main__":
    height1, width1 = 2,4
    height2 = width2 = 3
    rectangle1 = Rectangle((0, 0), size=(height1, width1))
    rectangle2 = Rectangle((1, 3), size=(height2, width2))
    
    assert rectangle1.area == height1 * width1
    assert rectangle2.area == height2 * width2
    
    #assert intersection_rectangles(rectangle1, rectangle2) == ((1,3), (2,4))
