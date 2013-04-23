__all__ = ['convex_hull_image', 'connected_component', 'convex_hull_object']

import numpy as np
from ._pnpoly import grid_points_inside_poly
from ._convex_hull import possible_hull
from .selem import square as sq
from skimage.morphology import label, dilation

def convex_hull_image(image):
    
    """Compute the convex hull image of a binary image.
    
    The convex hull is the set of pixels included in the smallest convex
    polygon that surround all white pixels in the input image.
    
    Parameters
    ----------
    image : ndarray
        Binary input image.  This array is cast to bool before processing.
    
    Returns
    -------
    hull : ndarray of bool
        Binary image with pixels in convex hull set to True.
    
    References
    ----------
    .. [1] http://blogs.mathworks.com/steve/2011/10/04/binary-image-convex-hull-algorithm-notes/
    
    """
    
    image = image.astype(bool)
    
    # Here we do an optimisation by choosing only pixels that are
    # the starting or ending pixel of a row or column.  This vastly
    # limits the number of coordinates to examine for the virtual
    # hull.
    coords = possible_hull(image.astype(np.uint8))
    N = len(coords)
    
    # Add a vertex for the middle of each pixel edge
    coords_corners = np.empty((N * 4, 2))
    for i, (x_offset, y_offset) in enumerate(zip((0, 0, -0.5, 0.5),
                                                 (-0.5, 0.5, 0, 0))):
        
        coords_corners[i * N:(i + 1) * N] = coords + [x_offset, y_offset]
        coords = coords_corners
    	try:
            from scipy.spatial import Delaunay
    	except ImportError:
            raise ImportError('Could not import scipy.spatial, only available in '
                              'scipy >= 0.9.')
    
    # Find the convex hull
    chull = Delaunay(coords).convex_hull
    v = coords[np.unique(chull)]
    
    # Sort vertices clock-wise
    v_centred = v - v.mean(axis=0)
    angles = np.arctan2(v_centred[:, 0], v_centred[:, 1])
    v = v[np.argsort(angles)]
    
    # For each pixel coordinate, check whether that pixel
    # lies inside the convex hull
    mask = grid_points_inside_poly(image.shape[:2], v)
    
    return mask

def connected_component(image, start_pixel_tuple):
    
    """Compute the connected object to a given starting pixel with 
    8-connectivity for a binary image
    
    The convex hull is the set of pixels included in the smallest convex
    polygon that surround all white pixels in the input image.
    
    Parameters
    ----------
    image : ndarray
        Binary input image.  
    
    start_pixel_tuple : tuple of int
        Of the form (x, y) which represents the index of the starting pixel
    
    Returns
    -------
    obj : ndarray of uint8
        Binary image with pixels in convex hull set to True.
    
    """
    
    next_im = np.zeros(image.shape, dtype=np.uint8)
    next_im[start_pixel_tuple] = 1
    start_im = np.zeros(image.shape)
    # Structuring element for Dilation: square of side 3 with all elements 1. 
    while not np.array_equal(start_im, next_im):
        start_im = next_im.copy()
        dilated_im = dilation(start_im, sq(3))
        next_im = dilated_im & image
    
    return next_im

def convex_hull_object(image, output_form=None):
    
    """Compute the convex hull image of individual objects in a binary image.
    
    The convex hull is the set of pixels included in the smallest convex
    polygon that surround all white pixels in the input image.
    
    Parameters
    ----------
    image : ndarray
        Binary input image.  
    
    output_form : string
        if 'single' then outputs a 3D array with separate convex hull computed 
        for individual objects, where the 3rd index is used to change the object
        Default is None, in which case it outputs the convex hull for all 
        objects individually as a single 2D array
    
    Returns
    -------
    hull : ndarray of bool
        Binary image with pixels in convex hull set to True.
    
    """
    
    # We add 1 to the output of label() so as to make the 
    # background 0 rather than -1
    
    (m, n) = image.shape
    convex_out = np.zeros((m, n), dtype=bool)
    labeled_im = label(image, neighbors=8, background=0) + 
    segmented_objs = np.zeros((m, n, labeled_im.max()), dtype=bool)
    convex_objs = np.zeros((m, n, labeled_im.max()), dtype=bool)
    
    for i in range(1, labeled_im.max()+1):
        
        start_pixel_tuple = tuple(transpose(np.where(labeled_im == i))[0])
        segmented_objs[:, :, i-1] = connected_component(image, start_pixel_tuple)
        convex_objs[:, :, i-1] = convex_hull_image(segmented_objs[:, :, i-1])
        convex_out |= convex_objs[:, :, i-1]
    
    if output_form is 'single':
        return convex_objs
    
    if output_form is None:
        return convex_out	
