import numpy as np


def integral_image(x):
    """Integral image / summed area table.

    The integral image contains the sum of all elements above and to the
    left of it, i.e.:

    .. math::

       S[m, n] = \sum_{i \leq m} \sum_{j \leq n} X[i, j]

    Parameters
    ----------
    x : ndarray
        Input image.

    Returns
    -------
    S : scalar value
        summed area table.

    References
    ----------
    .. [1] F.C. Crow, "Summed-area tables for texture mapping,"
           ACM SIGGRAPH Computer Graphics, vol. 18, 1984, pp. 207-212.

    """
    dim = len(x.shape)
    S = x
    for i in range(dim):
	S = S.cumsum(i)
    return S


def integrate(ii, start, end):
    """Use an integral image to integrate over a given window.

    Parameters
    ----------
    ii : ndarray
        Integral image.
    start : int or ndarray or list
        Top-left corner of block to be summed.
    end  : int or ndarray or list
        Bottom-right corner of block to be summed.

    Returns
    -------
    S : scalar 
        Integral (sum) over the given window.
    Notes
    -----
    Explination:
        For a 2D array say(10 x 10) intergral from start=(2,3) to end=(5,6) is
        #replace 'zero' elements from end -> permutation('00')
        +Intgral_array[5,6]
        #replace 'one' elements from end by 'start coorinate - 1' -> permutation('10','01')
        -(Integral_array[5,(3 - 1)] + integral_array[(2 - 1), 6])
        #replace 'two' elements from end by 'start coordinate - 1' -> permutation('11')
        +(Integral_array[(2-1),(3-1)])
    """
    #make sure start and end both are arrays	
    start = np.asarray(start)
    end = np.asarray(end)

    if(np.any(start < 0) or np.any(end < 0)):
	raise IndexingError('cordinates must be non negative')

    if(np.any((end - start) < 0)):
	raise Error('end coordinates must be larger or equal to start')

    dim = len(ii.shape)  #No. of dimensions of input nd-array 
    S = 0
    bit_perm = 2**dim  #bit_perm is the total number of elements in expression of S
    width = len(bin(bit_perm-1)[2:])

    for i in range(bit_perm):  #for all permutations
        #generate boolean array corresponding to permutation eg [True, False] for '10'		      
        binary = bin(i)[2:].zfill(width)
	bool_mask = [bit=='1' for bit in binary]
        
        sign = (-1)**sum(bool_mask)  #determine sign of permutation
        bad = np.any(((start - 1)*bool_mask) < 0)
        if(bad):
	    continue
	 
        corner_point = (end * (np.invert(bool_mask))) + ((start - 1) * bool_mask)
        
        S += sign*ii[tuple(corner_point)]
    return S
