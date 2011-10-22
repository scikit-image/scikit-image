"""Use an iterative thinning algorithm to find the skeletons of binary 
objects in an image.

"""

import numpy as np
from scipy import ndimage

from _cpmorphology2 import skeletonize_loop, table_lookup_index
from _cpmorphology2 import extract_from_image_lookup, \
            prepare_for_index_lookup, index_lookup

# --------- Skeletonization by morphological thinning ---------

def skeletonize(image):
    """Return the skeleton of a binary image.
    
    Thinning is used to reduce each connected component in a binary image
    to a single-pixel wide skeleton. 
     
    Parameters
    ----------
    image : numpy.ndarray 
        A binary image containing the objects to be skeletonized. '1' 
        represents foreground, and '0' represents background. It 
        also accepts arrays of boolean values where True is foreground.
    
    Returns
    -------
    skeleton : ndarray
        A matrix containing the thinned image.
    
    See also
    --------
    medial_axis

    Notes
    -----
    The algorithm [1] works by making successive passes of the image, 
    removing pixels on object borders. This continues until no
    more pixels can be removed.  The image is correlated with a
    mask that assigns each pixel a number in the range [0...255]
    corresponding to each possible pattern of its 8 neighbouring
    pixels. A look up table is then used to assign the pixels a
    value of 0, 1, 2 or 3, which are selectively removed during
    the iterations.
    
    Note that this algorithm will give different results than a 
    medial axis transform, which is also often referred to as
    "skeletonization".   
   
    References
    ----------
    .. [1] A fast parallel algorithm for thinning digital patterns, 
       T. Y. ZHANG and C. Y. SUEN, Communications of the ACM,
       March 1984, Volume 27, Number 3

    
    Examples
    --------
    >>> X, Y = np.ogrid[0:9, 0:9]
    >>> ellipse = (1./3 * (X - 4)**2 + (Y - 4)**2 < 3**2).astype(np.uint8)
    >>> ellipse
    array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 1, 1, 0, 0],
           [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=uint8)
    >>> skel = skeletonize(ellipse)
    >>> skel
    array([[0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
           
    """
    # look up table - there is one entry for each of the 2^8=256 possible
    # combinations of 8 binary neighbours. 1's, 2's and 3's are candidates
    # for removal at each iteration of the algorithm.
    lut = [ 0,0,0,1,0,0,1,3,0,0,3,1,1,0,1,3,0,0,0,0,0,0,0,0,2,0,2,0,3,0,3,3,
            0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,2,0,0,0,3,0,2,2,
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            2,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,3,0,0,0,0,0,0,0,3,0,0,0,3,0,2,0,
            0,1,3,1,0,0,1,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
            3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            2,3,1,3,0,0,1,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            2,3,0,1,0,0,0,1,0,0,0,0,0,0,0,0,3,3,0,1,0,0,0,0,2,2,0,0,2,0,0,0]

    # convert to unsigned int (this should work for boolean values)
    skeleton = np.array(image).astype(np.uint8)

    # check some properties of the input image:
    #  - 2D
    #  - binary image with only 0's and 1's 
    if skeleton.ndim != 2:
        raise ValueError('Skeletonize requires a 2D array')    
    if not np.all(np.in1d(skeleton.flat, (0, 1))):
        raise ValueError('Image contains values other than 0 and 1')
    
    # create the mask that will assign a unique value based on the
    #  arrangement of neighbouring pixels
    mask = np.array([[  1,  2,  4],
                     [128,  0,  8],
                     [ 64, 32, 16]], np.uint8)

    pixelRemoved = True
    while pixelRemoved:
        pixelRemoved = False;

        # assign each pixel a unique value based on its foreground neighbours
        neighbours = ndimage.correlate(skeleton, mask, mode='constant')
        
        # ignore background
        neighbours *= skeleton
        
        # use LUT to categorize each foreground pixel as a 0, 1, 2 or 3
        codes = np.take(lut, neighbours)
        
        # pass 1 - remove the 1's and 3's
        code_mask = (codes == 1)
        if np.any(code_mask): 
            pixelRemoved = True
            skeleton[code_mask] = 0
        code_mask = (codes == 3)
        if np.any(code_mask): 
            pixelRemoved = True
            skeleton[code_mask] = 0

        # pass 2 - remove the 2's and 3's
        neighbours = ndimage.correlate(skeleton, mask, mode='constant')
        neighbours *= skeleton
        codes = np.take(lut, neighbours)
        code_mask = (codes == 2)
        if np.any(code_mask): 
            pixelRemoved = True
            skeleton[code_mask] = 0
        code_mask = (codes == 3)
        if np.any(code_mask): 
            pixelRemoved = True
            skeleton[code_mask] = 0

    return skeleton

# --------- Skeletonization by medial axis transform --------

eight_connect = ndimage.generate_binary_structure(2, 2)


def medial_axis(image, mask=None, return_distance=False):
    """
    Compute the medial axis transform of a binary image

    Parameters
    ----------

    image: binary ndarray 
    
    mask: binary ndarray, optional
        If a mask is given, only those elements with a true value in `mask`
        are used for computing the medial axis.

    return_distance; bool, optional
        If true, the distance transform is returned as well as the skeleton.

    Returns
    -------

    out: ndarray of bools
        Medial axis transform of the image

    dist: ndarray of ints
        Distance transform of the image (only returned if `return_distance`
        is True)
   
    See also
    --------
    skeletonize 

    Notes
    -----
    This algorithm computes the medial axis transform of an image
    as the ridges of its distance transform. First, the distance transform
    is computed, then the foreground (value of 1) points are ordered by
    the distance transform. In order to reduce the image to its skeleton, 
    a point is removed if it has more than one neighbor and if removing it
    does not change the Euler number (the connectivity).

    Examples
    --------
    >>> square = np.zeros((7, 7), dtype=np.uint8)
    >>> square[1:-1, 2:-2] = 1
    >>> square
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> morphology.medial_axis(square).astype(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    """
    global eight_connect
    if mask is None:
        masked_image = image.astype(np.bool)
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    #
    # Lookup table - start with only positive pixels.
    # Keep if # pixels in neighborhood is 2 or less
    # Keep if removing the pixel results in a different connectivity
    # table is independent of image
    table = (_make_table(True, 
                        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], bool),
                        np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], bool)) &
            (np.array([ndimage.label(_pattern_of(index), eight_connect)[1] !=
                        ndimage.label(_pattern_of(index & ~ 2**4),
                                    eight_connect)[1]
                        for index in range(512)]) |
        np.array([np.sum(_pattern_of(index)) < 3 for index in range(512)])))
    distance = ndimage.distance_transform_edt(masked_image)
    if return_distance:
        store_distance = distance.copy()
    #
        # The processing order along the edge is critical to the shape of the
    # resulting skeleton: if you process a corner first, that corner will
    # be eroded and the skeleton will miss the arm from that corner. Pixels
    # with fewer neighbors are more "cornery" and should be processed last.
    #
    cornerness_table = np.array([9 - np.sum(_pattern_of(index))
                                 for index in range(512)])
    corner_score = _table_lookup(masked_image, cornerness_table, False, 1)
    i, j = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    result = masked_image.copy()
    distance = distance[result]
    i = np.ascontiguousarray(i[result], np.int32)
    j = np.ascontiguousarray(j[result], np.int32)
    result = np.ascontiguousarray(result, np.uint8)
    #
    # We use a random # for tiebreaking. Assign each pixel in the image a
    # predictable, random # so that masking doesn't affect arbitrary choices
    # of skeletons
    #
    # Why fix the seed? Should we pass a random number generator instead?
    np.random.seed(0)
    tiebreaker = np.random.permutation(np.arange(masked_image.sum()))
    order = np.lexsort((tiebreaker,
                        corner_score[masked_image],
                        distance))
    order = np.ascontiguousarray(order, np.int32)
    table = np.ascontiguousarray(table, np.uint8)
    # Remove pixels not belonging to the medial axis
    skeletonize_loop(result, i, j, order, table)

    result = result.astype(bool)
    if not mask is None:
        result[~mask] = image[~mask]
    if return_distance:
        return result, store_distance
    else:
        return result

def _pattern_of(index):
    """
    Return the pattern represented by an index value
    Byte decomposition of index
    """
    return np.array([[index & 2**0,index & 2**1,index & 2**2],
                     [index & 2**3,index & 2**4,index & 2**5],
                     [index & 2**6,index & 2**7,index & 2**8]], bool)


def _table_lookup(image, table, border_value, iterations = None):
    """
    Perform a morphological transform on an image, directed by its 
    neighbors
    
    Parameters
    ----------
    image - a binary image
    table - a 512-element table giving the transform of each pixel given
            the values of that pixel and its 8-connected neighbors.
    border_value - the value of pixels beyond the border of the image.
                   This should test as True or False.
    
    Returns
    -------
    result: ndarray of same shape as `image`
        Transformed image

    Notes
    -----
    The pixels are numbered like this:
    
    0 1 2
    3 4 5
    6 7 8
    The index at a pixel is the sum of 2**<pixel-number> for pixels
    that evaluate to true. 
    """
    #
    # Test for a table that never transforms a zero into a one:
    #
    center_is_zero = np.array([(x & 2**4) == 0 for x in range(2**9)])
    use_index_trick = False
    if (not np.any(table[center_is_zero]) and
        (np.issubdtype(image.dtype, bool) or np.issubdtype(image.dtype, int))):
        # Use the index trick
        use_index_trick = True
        invert = False
    elif (np.all(table[~center_is_zero]) and np.issubdtype(image.dtype, bool)):
        # All ones stay ones, invert the table and the image and do the trick
        use_index_trick = True
        invert = True
        image = ~ image
        # table index 0 -> 511 and the output is reversed
        table = ~ table[511-np.arange(512)]
        border_value = not border_value
    if use_index_trick:
        orig_image = image
        index_i, index_j, image = prepare_for_index_lookup(image, border_value)
        index_i, index_j = index_lookup(index_i, index_j,
                                        image, table, iterations)
        image = extract_from_image_lookup(orig_image, index_i, index_j)
        if invert:
            image = ~ image
        return image
    print(use_index_trick)
    counter = 0
    while counter != iterations:
        counter += 1
        #
        # We accumulate into the indexer to get the index into the table
        # at each point in the image
        #
        if image.shape[0] < 3 or image.shape[1] < 3:
            image = image.astype(bool)
            indexer = np.zeros(image.shape,int)
            indexer[1:, 1:]   += image[:-1, :-1] * 2**0
            indexer[1:, :]    += image[:-1, :] * 2**1
            indexer[1:, :-1]  += image[:-1, 1:] * 2**2
            
            indexer[:, 1:]    += image[:, :-1] * 2**3
            indexer[:, :]     += image[:, :] * 2**4
            indexer[:, :-1]   += image[:, 1:] * 2**5
        
            indexer[:-1, 1:]  += image[1:, :-1] * 2**6
            indexer[:-1, :]   += image[1:, :] * 2**7
            indexer[:-1, :-1] += image[1:, 1:] * 2**8
        else:
            indexer = table_lookup_index(np.ascontiguousarray(image, np.uint8))
        if border_value:
            indexer[0,:]   |= 2**0 + 2**1 + 2**2
            indexer[-1,:]  |= 2**6 + 2**7 + 2**8
            indexer[:,0]   |= 2**0 + 2**3 + 2**6
            indexer[:,-1]  |= 2**2 + 2**5 + 2**8
        new_image = table[indexer]
        if np.all(new_image == image):
            break
        image = new_image
    return image

def _make_table(value, pattern, care=np.ones((3,3),bool)):
    '''Return a table suitable for table_lookup
    
    value - set all table entries matching "pattern" to "value", all others
            to not "value"
    pattern - a 3x3 boolean array with the pattern to match
    care    - a 3x3 boolean array where each value is true if the pattern
              must match at that position and false if we don't care if
              the pattern matches at that position.
    '''
    def fn(index, p, i, j):
        '''Return true if bit position "p" in index matches pattern'''
        return ((((index & 2**p) > 0) == pattern[i, j]) or not care[i, j])
    return np.array([value     
                     if (fn(i, 0, 0, 0) and fn(i, 1, 0, 1) and fn(i, 2, 0, 2) 
                     and fn(i, 3, 1, 0) and fn(i, 4, 1, 1) and fn(i, 5, 1, 2) 
                     and fn(i, 6, 2, 0) and fn(i, 7, 2, 1) and fn(i, 8, 2, 2))
                     else not value
                     for i in range(512)], bool)


