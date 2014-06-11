"""
Algorithms for computing the skeleton of a binary image
"""

import numpy as np
from scipy import ndimage

from ._skeletonize_cy import _skeletonize_loop, _table_lookup_index

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
    >>> skel.astype(np.uint8)
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
    lut = [0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0,
           0, 0, 2, 0, 2, 0, 3, 0, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0,
           0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 3, 1,
           0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 1, 3, 0, 0,
           1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3,
           0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0, 0]

    # convert to unsigned int (this should work for boolean values)
    skeleton = image.astype(np.uint8)

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

    pixel_removed = True
    while pixel_removed:
        pixel_removed = False

        # assign each pixel a unique value based on its foreground neighbours
        neighbours = ndimage.correlate(skeleton, mask, mode='constant')

        # ignore background
        neighbours *= skeleton

        # use LUT to categorize each foreground pixel as a 0, 1, 2 or 3
        codes = np.take(lut, neighbours)

        # pass 1 - remove the 1's and 3's
        code_mask = (codes == 1)
        if np.any(code_mask):
            pixel_removed = True
            skeleton[code_mask] = 0
        code_mask = (codes == 3)
        if np.any(code_mask):
            pixel_removed = True
            skeleton[code_mask] = 0

        # pass 2 - remove the 2's and 3's
        neighbours = ndimage.correlate(skeleton, mask, mode='constant')
        neighbours *= skeleton
        codes = np.take(lut, neighbours)
        code_mask = (codes == 2)
        if np.any(code_mask):
            pixel_removed = True
            skeleton[code_mask] = 0
        code_mask = (codes == 3)
        if np.any(code_mask):
            pixel_removed = True
            skeleton[code_mask] = 0

    return skeleton.astype(bool)

# --------- Skeletonization by medial axis transform --------

_eight_connect = ndimage.generate_binary_structure(2, 2)


def medial_axis(image, mask=None, return_distance=False):
    """
    Compute the medial axis transform of a binary image

    Parameters
    ----------
    image : binary ndarray, shape (M, N)
        The image of the shape to be skeletonized.
    mask : binary ndarray, shape (M, N), optional
        If a mask is given, only those elements in `image` with a true
        value in `mask` are used for computing the medial axis.
    return_distance : bool, optional
        If true, the distance transform is returned as well as the skeleton.

    Returns
    -------
    out : ndarray of bools
        Medial axis transform of the image
    dist : ndarray of ints, optional
        Distance transform of the image (only returned if `return_distance`
        is True)

    See also
    --------
    skeletonize

    Notes
    -----
    This algorithm computes the medial axis transform of an image
    as the ridges of its distance transform.

    The different steps of the algorithm are as follows
     * A lookup table is used, that assigns 0 or 1 to each configuration of
       the 3x3 binary square, whether the central pixel should be removed
       or kept. We want a point to be removed if it has more than one neighbor
       and if removing it does not change the number of connected components.

     * The distance transform to the background is computed, as well as
       the cornerness of the pixel.

     * The foreground (value of 1) points are ordered by
       the distance transform, then the cornerness.

     * A cython function is called to reduce the image to its skeleton. It
       processes pixels in the order determined at the previous step, and
       removes or maintains a pixel according to the lookup table. Because
       of the ordering, it is possible to process all pixels in only one
       pass.

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
    >>> medial_axis(square).astype(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)

    """
    global _eight_connect
    if mask is None:
        masked_image = image.astype(np.bool)
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    #
    # Build lookup table - three conditions
    # 1. Keep only positive pixels (center_is_foreground array).
    # AND
    # 2. Keep if removing the pixel results in a different connectivity
    # (if the number of connected components is different with and
    # without the central pixel)
    # OR
    # 3. Keep if # pixels in neighbourhood is 2 or less
    # Note that table is independent of image
    center_is_foreground = (np.arange(512) & 2**4).astype(bool)
    table = (center_is_foreground  # condition 1.
                &
            (np.array([ndimage.label(_pattern_of(index), _eight_connect)[1] !=
                        ndimage.label(_pattern_of(index & ~ 2**4),
                                    _eight_connect)[1]
                        for index in range(512)])  # condition 2
                |
        np.array([np.sum(_pattern_of(index)) < 3 for index in range(512)]))
        # condition 3
            )

    # Build distance transform
    distance = ndimage.distance_transform_edt(masked_image)
    if return_distance:
        store_distance = distance.copy()

    # Corners
    # The processing order along the edge is critical to the shape of the
    # resulting skeleton: if you process a corner first, that corner will
    # be eroded and the skeleton will miss the arm from that corner. Pixels
    # with fewer neighbors are more "cornery" and should be processed last.
    # We use a cornerness_table lookup table where the score of a
    # configuration is the number of background (0-value) pixels in the
    # 3x3 neighbourhood
    cornerness_table = np.array([9 - np.sum(_pattern_of(index))
                                 for index in range(512)])
    corner_score = _table_lookup(masked_image, cornerness_table)

    # Define arrays for inner loop
    i, j = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    result = masked_image.copy()
    distance = distance[result]
    i = np.ascontiguousarray(i[result], dtype=np.intp)
    j = np.ascontiguousarray(j[result], dtype=np.intp)
    result = np.ascontiguousarray(result, np.uint8)

    # Determine the order in which pixels are processed.
    # We use a random # for tiebreaking. Assign each pixel in the image a
    # predictable, random # so that masking doesn't affect arbitrary choices
    # of skeletons
    #
    generator = np.random.RandomState(0)
    tiebreaker = generator.permutation(np.arange(masked_image.sum()))
    order = np.lexsort((tiebreaker,
                        corner_score[masked_image],
                        distance))
    order = np.ascontiguousarray(order, dtype=np.int32)

    table = np.ascontiguousarray(table, dtype=np.uint8)
    # Remove pixels not belonging to the medial axis
    _skeletonize_loop(result, i, j, order, table)

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
    return np.array([[index & 2**0, index & 2**1, index & 2**2],
                     [index & 2**3, index & 2**4, index & 2**5],
                     [index & 2**6, index & 2**7, index & 2**8]], bool)


def _table_lookup(image, table):
    """
    Perform a morphological transform on an image, directed by its
    neighbors

    Parameters
    ----------
    image : ndarray
        A binary image
    table : ndarray
        A 512-element table giving the transform of each pixel given
        the values of that pixel and its 8-connected neighbors.
    border_value : bool
        The value of pixels beyond the border of the image.

    Returns
    -------
    result : ndarray of same shape as `image`
        Transformed image

    Notes
    -----
    The pixels are numbered like this::


      0 1 2
      3 4 5
      6 7 8

    The index at a pixel is the sum of 2**<pixel-number> for pixels
    that evaluate to true.
    """
    #
    # We accumulate into the indexer to get the index into the table
    # at each point in the image
    #
    if image.shape[0] < 3 or image.shape[1] < 3:
        image = image.astype(bool)
        indexer = np.zeros(image.shape, int)
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
        indexer = _table_lookup_index(np.ascontiguousarray(image, np.uint8))
    image = table[indexer]
    return image
