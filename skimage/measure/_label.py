import numpy as np

from ._ccomp import label as _label
from ._ccomp import relabel_array as _relabel_array

__all__ = ['label', 'label_match']

def label(input, neighbors=8, background=None, return_num=False):
    return _label(input, neighbors, background, return_num)

label.__doc__ = _label.__doc__

def _get_duplicate_hash(lab1, lab2, label_1, label_2, pair, background=-1):
    """
    Find any labels that overlap more than one label in the other image, and 
    remove the label from the label list and create a dict that maps the 
    label to be removed to the background value.
    
    Parameters
    ----------
    lab1: array
        The label numbers of all intersecting labels in image 1
    
    lab2: array
        The label numbers of all intersecting labels in image 2

    label_1, label_2: ndarray
        Two labelled arrays, not relablled.
    
    pair: array
        The single array with both label arrays encoded in each pixel "xxxyyy"
    
    background: int
        The value to use of a non-labeled pixel

    Returns
    --------
    
    lab1, lab2: ndarray
        Input arrays with smallest overlaps removed
    
    hash1, hash2: dict
        Mapping of duplicates to background, so that rebel will remove them.
    """

    # Get the indexes for all the duplicate overlaps
    dup1 = np.in1d(lab1,(np.bincount(lab1) > 1).nonzero()[0])
    dup2 = np.in1d(lab2,(np.bincount(lab2) > 1).nonzero()[0])
    
    # Get the sorted values of the overlaps and the indicies to reconstruct the array
    d, i = np.unique(pair[np.logical_and(label_1, label_2)], return_inverse=True)
    # Sort the indicies and count them to compute overalap area for each number in the "d" array
    i.sort()
    overlap_area = np.bincount(i)
    
    # Create a boolean array the same shape as the 1D array of overlap numbers
    ii1 = np.zeros(lab1.shape, dtype=bool)
    # Make it true where the overlaps are
    ii1 = np.logical_or(ii1, dup1)
    # If we have overlaps
    if dup1.nonzero()[0].size:
        # Set the largest overlap to False, so we leave it alone.
        ii1[np.argmax(overlap_area[dup1])] = False
    # Create a hash table to map the overlaps where the mask is true to the background value
    hash2 = dict(zip(lab2[ii1], np.zeros(lab1[ii1].size)+background))
    
    ii2 = np.zeros(lab2.shape, dtype=bool)
    ii2 = np.logical_or(ii2, dup2)
    if dup2.nonzero()[0].size:
        ii2[np.argmax(overlap_area[dup2])] = False
    hash1 = dict(zip(lab1[ii2], np.zeros(lab1[ii2].size)+background))
    
    # Remove the overlaps from the label arrays
    ii = np.logical_or(ii1, ii2)
    lab1 = np.delete(lab1, ii.nonzero())
    lab2 = np.delete(lab2, ii.nonzero())
    
    return lab1, lab2, hash1, hash2

def _find_overlapping_labels(label_1, label_2):
    """
    Find the overlapping labels in two images and return two 1D numpy arrays
    mapping the labels in the two images together.
    
    Parameters
    ----------
    label_1, label_2: ndarray
        Two labelled arrays.
    
    Returns
    -------
    lab1, lab2: ndarray
        Two 1D arrays of the same length mapping the labels in both images.
    
    pair: ndarray
        An image containing both labels in the form xxxyyy. Where x is the 
        label from image 1 and y is the label from image 2. The amount of 
        digits required to make the value unique is automatically calculated.
    """
    # Calulate the number to multiply the first array by so
    # that the two sets of labels are seperated by 0s
    shift = (10**(np.floor(np.log10(np.max([np.max(label_1),
                                            np.max(label_2)])))+1)).astype(int)

    # re-make label_1 and add it to label_2 to create one array with both numbers
    label_1_2 = label_1.copy() * shift
    pair = label_1_2 + label_2

    # Extract the numbers at the point where the arrays overlap
    doubles = np.unique(pair[np.logical_and(label_1, label_2)])

    # Get the labels of the overlaps for each input image
    lab1 = doubles // shift
    lab2 = doubles % shift
    
    return lab1, lab2, pair

def label_match(label_1, label_2, relabel=True, remove_nonoverlap=False,
                remove_duplicates=False, background=-1):
    """
    Take two labelled arrays and find the labels that overlap in both images.
    
    Parameters
    ----------
    label_1, label_2: ndarray
        Two labelled arrays of same shape.
    
    relabel: bool
        If true then relabel the input arrays so that if the labels that overlap,
        they share the same label number.
    
    remove_nonoverlap: bool or int
        relabel the non overlapping regions to the background (if True) or
        relabel sequentially higher than remove_overlap if an integer.
        This allows you to keep the labels unique, because non overlapping 
        regions are relabelled to a different number than overlapping ones.
    
    remove_duplicates: bool
        If one label in one image overlaps with more than one  in the other,
        remove the labels in the other image that have the smallest overlap.
    
    background: int
        The number to use where no label is present.

    Returns
    -------
    label_1, label_2: ndarray
        Modified versions of the input arrays with the labels changed.
    """

    label_1 = np.asanyarray(label_1)
    label_2 = np.asanyarray(label_2)

    if label_1.shape != label_2.shape:
        raise TypeError("The two labelled arrays must have the same dimensions")

    lab1, lab2, pair = _find_overlapping_labels(label_1, label_2)

    # Ravel out the images into a 1D view
    label_1_rav = label_1.ravel()
    label_2_rav = label_2.ravel()
    
    # Map hash
    hash1 = {}
    hash2 = {}
    
    if isinstance(remove_nonoverlap, bool):
        if remove_nonoverlap:
            # Set non overlapping labels in the array to background
            label_1_rav[np.logical_not(np.in1d(label_1_rav, lab1))] = background
            label_2_rav[np.logical_not(np.in1d(label_2_rav, lab2))] = background

    elif isinstance(remove_nonoverlap, int):
        # Compute the 1D intersection between the known overlapping labels and the array
        # and not the background
        noover_1 = label_1_rav[np.logical_and(np.logical_not(np.in1d(label_1_rav, lab1)),
                                              np.logical_not(label_1_rav == background))]
        noover_2 = label_2_rav[np.logical_and(np.logical_not(np.in1d(label_2_rav, lab2)),
                                              np.logical_not(label_2_rav == background))]

        # Add the non-overlapping relabel to the dict
        hasha = dict(zip(noover_1, range(remove_nonoverlap+1, len(noover_1)+1)))
        hashb = dict(zip(noover_2, range(remove_nonoverlap+1, len(noover_2)+1)))
        
        hash1.update(hasha)
        hash2.update(hashb)
    
    else:
        raise TypeError("remove_overlap must be boolean or integer type")

    # This routine creates a dictionary of mappings between current label values
    # and desired ones which is then passed to the relabel routine.
    if remove_duplicates:
        # Calculate any duplicates and remove them
        lab1, lab2, hasha, hashb = _get_duplicate_hash(lab1, lab2, label_1, label_2, 
                                                       pair, background=background)

        hash1.update(hasha)
        hash2.update(hashb)

    if relabel:
        #Create hashtable mappings of the current labels to the new labels
        hasha = dict(zip(lab1, range(background+1,len(lab1)+1)))
        hashb = dict(zip(lab2, range(background+1,len(lab2)+1)))
    
        hash1.update(hasha)
        hash2.update(hashb)
    
    if hash1 != {} and hash2 != {}:
        # Call the relabel function
        # (that is still working on views to label_1 and label_2)
        label_1_rav = _relabel_array(hash1, label_1_rav)
        label_2_rav = _relabel_array(hash2, label_2_rav)

    #Return the labelled and filtered arrays in their 2D forms
    return label_1, label_2