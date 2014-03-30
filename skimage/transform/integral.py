import numpy as np


def integral_image(img):
    """Integral image / summed area table.

    The integral image contains the sum of all elements above and to the
    left of it, i.e.:

    .. math::

       S[m, n] = \sum_{i \leq m} \sum_{j \leq n} X[i, j]

    Parameters
    ----------
    img : ndarray
        Input image.

    Returns
    -------
    S : ndarray
        Integral image/summed area table of same shape as input image.

    References
    ----------
    .. [1] F.C. Crow, "Summed-area tables for texture mapping,"
           ACM SIGGRAPH Computer Graphics, vol. 18, 1984, pp. 207-212.

    """
    S = img
    for i in range(img.ndim):
        S = S.cumsum(axis=i)
    return S


def integrate(ii, start, end, *args):
    """Use an integral image to integrate over a given window.

    Parameters
    ----------
    ii : ndarray
        Integral image.
    start : tuple of length equal to dimension of ii
        Coordinates of top left corner of window(s).
        For multiple windows each coordinate should be a list
	using same format as numpy multi-indexing conventions.
    end : tuple of length equal to dimension of ii
        Coordinates of bottom right corner of window(s).
        For multiple windows each coordinate should be a list
	using same format as numpy multi-indexing conventions.
    args: optional
        For backward compatibility with versions prior to 0.10
	The earlier function signature was integrate(ii, r0, c0, r1, c1),
        where r0, c0 are int(lists) specifying start coordinates 
        of window(s) to be integrated and r1, c1 the end coordinates. 
       

    Returns
    -------
    S : scalar or ndarray
        Integral (sum) over the given window(s).


    Examples
    --------
    >>> arr = np.ones((5,6), dtype=np.float)
    >>> ii = integral_image(arr)
    >>> print(integrate(ii,(1,0), (1,2)))  # sum from (1,0) -> (1,2)
    [ 3.]
    >>> print(integrate(ii,(3,3), (4,5)))  # sum form (3,3) -> (4,5)
    [ 6.]
    >>> print(integrate(ii,([1,3], [0,3]), ([1,4], [2,5])))  # sum from (1,0) -> (1,2) and (3,3) -> (4,5) 
    [ 3.  6.]
    >>> print(integrate(ii, [1,3], [0,3], [1,4], [2,5]))  # deprecated usage
    [ 3.  6.]
    """
    rows = 1
    # handle input from new input format
    if(len(args) == 0):
        if(not(isinstance(start[0], int))):
            rows = len(start[0])
        start = np.array(start).T
        end = np.array(end).T
    # handle deprecated input format
    else:
        if(not(isinstance(start, int))):
            rows = len(start)
        args = (start , end) + args
        start = np.array(args[:len(args)/2]).T
        end = np.array(args[len(args)/2:]).T


    total_shape = ii.shape
    total_shape = np.tile(total_shape, [rows, 1])
    # take care of negative coordinates

    start_negatives  = start < 0
    end_negatives = end < 0
    start = (start + total_shape) * start_negatives + \
            start * np.invert(start_negatives)
    end = (end + total_shape) * end_negatives + \
          end * np.invert(end_negatives)


    if(np.any((end - start) < 0)):
        raise IndexError('end coordinates must be greater or equal to start')


    S = np.zeros(rows)
    bit_perm = 2**(ii.ndim)  # total number of elements in expression of S
    width = len(bin(bit_perm - 1)[2:])

    for i in range(bit_perm):  # for all permutations
        # boolean permutation array eg [True, False] for '10'
        binary = bin(i)[2:].zfill(width)
        bool_mask = [bit == '1' for bit in binary]
        
        sign = (-1)**sum(bool_mask)  # determine sign of permutation
        
        bad = [np.any(((start[r] - 1) * bool_mask) < 0)
               for r in range(rows)] # find out bad start rows

        corner_points = (end * (np.invert(bool_mask))) + \
                         ((start - 1) * bool_mask) # find corner for each row
        
        S += [sign * ii[tuple(corner_points[r])] if(bad[r] == False) else 0
              for r in range(rows)] # add only good rows
    return S
