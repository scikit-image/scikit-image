"""
:author: Damian Eads, 2009
:license: modified BSD
"""

import numpy as np

def square(width, dtype=np.uint8):
    """
    Generates a flat, square-shaped structuring element. Every pixel
    along the perimeter has a chessboard distance no greater than radius
    (radius=floor(width/2)) pixels.

    Parameters
    ----------
    width : int
       The width and height of the square

    Additional Parameters
    ---------------------
    
    dtype : string
       The data type of the structuring element.

    Returns
    -------
    selem : ndarray
       A structuring element consisting only of ones, i.e. every
       pixel belongs to the neighborhood.
    """
    return np.ones((width, width), dtype=dtype)

def rectangle(width, height, dtype=np.uint8):
    """
    Generates a flat, rectangular-shaped structuring element of a
    given width and height. Every pixel in the rectangle belongs
    to the neighboorhood.

    Parameters
    ----------
    width : int
       The width of the rectangle

    height : int
       The height of the rectangle

    Additional Parameters
    ---------------------
    
    dtype : string
       The data type of the structuring element.

    Returns
    -------
    selem : ndarray

       A structuring element consisting only of ones, i.e. every
       pixel belongs to the neighborhood.
    """
    return np.ones((width, height), dtype=dtype)

def diamond(radius, dtype=np.uint8):
    """
    Generates a flat, diamond-shaped structuring element of a given
    radius.  A pixel is part of the neighborhood (i.e. labeled 1) iff
    the city block/manhattan distance between it and the center of the
    neighborhood is no greater than radius.

    Parameters
    ----------
    radius : string
       The radius of the disk-shaped structuring element.

    dtype : string
       The data type of the structuring element.

    Returns
    -------
    
    selem : ndarray
       The structuring element where elements of the neighborhood
       are 1 and 0 otherwise.
    """
    half = radius
    (I, J) = np.meshgrid(xrange(0, radius*2+1), xrange(0, radius*2+1))
    s = np.abs(I-half)+np.abs(J-half)
    return np.array(s <= radius, dtype=dtype)
    
def disk(radius, N=0, dtype=np.uint8):
    """
    Generates a flat, disk-shaped structuring element of a given radius.
    A pixel is within the neighborhood iff the euclidean distance between
    it and the origin is no greater than a radius.
    
    Parameters
    ----------
    radius : string
       The radius of the disk-shaped structuring element.

    dtype : string
       The data type of the structuring element.

    Returns
    -------
    selem : ndarray
       The structuring element where elements of the neighborhood
       are 1 and 0 otherwise.          
    """
    half = radius
    (I, J) = np.meshgrid(xrange(0, radius*2+1), xrange(0, radius*2+1))
    if N == 0:
        s = (I-half)**2.+(J-half)**2.
        #print s
    else:
        raise NotImplementedError("""
        scikits.image.morphology.disk: approximations not implemented.

        Try N=0 for now.
        """)
    return np.array(s <= radius * radius, dtype=dtype)

def ellipse(size, angle, ratio=0.5, dtype=np.uint8):
    """
    Generates an elliptically-shaped structuring element of a given
    angle, ratio, and kernel size.
    
    Parameters
    ----------
    size : int
       The half-width of the kernel. The kernel size is 2*size+1.

    angle : float
       The angle of rotation of the ellipse in radians.
    
    ratio : float
       The aspect ratio of the ellipse.
   
    dtype : string
       The data type of the structuring element.
         
    Returns
    -------
    selem : ndarray
       The structuring element where elements of the neighborhood
       are 1 and 0 otherwise.
    """
    structure = np.zeros((2*size+1, 2*size+1), dtype=dtype)
    
    a = np.matrix([np.cos(angle), np.sin(angle)])
    b = np.matrix([-np.sin(angle), np.cos(angle)])
    aspect = a.T * a +  b.T * b / ratio

    for y in xrange(-size, size+ 1):
        for x in xrange(-size, size + 1):
            i = x+size
            j = y+size
            
            v = np.matrix([x,y], dtype='f') / float(size)
            n = v * aspect * v.T

            if n < 1:
                structure[i, j] = 1
    return structure

def strel(shape='disk', N=0, radius=3, width=3, height=3, angle=0., length=3,
          dtype=np.uint8, out=None):
    """
    Generates a structuring element for greyscale or binary morphology.
    The interface of this function is similar to MATLAB(TM)'s strel function.

    Parameters
    ----------
    shape : string
       A string identifier for the shape of the structuring element,
       which can be any of the following: 'arbitrary', 'ball',
      'diamond', 'disk', 'pair', 'rectangle', 'square'.

    N : int
       When non-zero, the number of lines to approximate the
       structuring element. (not implemented)

    radius : int
       The radius for disk or diamond structuring elements.

    width : int
       The height for square, ball or rectangle-shaped structuring elements.
      
    height : int
       The height for ball or rectangle-shaped structuring elements.

    size : int
       The half-width of an elliptical structuring element.

    aspect : float
       The aspect ratio of an ellipse.

    Returns
    -------
    neighborhood : ndarray
       The structuring element.
    """
    shape = shape.lower().strip()
    if shape == 'disk':
        return disk(radius, dtype=dtype)
    elif shape == 'diamond':
        return diamond(radius, dtype=dtype)
    elif shape == 'square':
        return square(width, dtype=dtype)
    elif shape == 'rectangle':
        return rectangle(width, height, dtype=dtype)
    elif shape == 'ellipse':
        return ellipse(size, angle, ratio, dtype=dtype)
    else:
        raise ValueError("Unknown structuring element type '%s'" % shape)
