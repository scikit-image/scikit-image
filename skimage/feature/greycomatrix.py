"""
Compute grey level co-occurrence matrices (GLCM) to characterize
image textures.
"""

import numpy as np
import skimage.util


def glcm(image, distances, angles, levels=256, symmetric=False,
         normal=False):
    """Calculate the grey-level co-occurrence matrix of a grey-level
    image.

    A grey level co-occurence matrix is a histogram of co-occuring
    greyscale values at a given offset over an image. It can be used to
    extract features from textured areas of an image.

    Parameters
    ----------
    image : (M,N) ndarray
        Input image. The input image is converted to the uint8 data
        type.
    distances : (K,) ndarray
        Histogram distance offsets
    angles : (L,) ndarray
        Histogram angles in radians
    levels : int
        The input image should contain integers in [0, levels-1],
        where levels indicate the number of grey-levels counted
        (typically 256 for an 8-bit image).
    symmetric : bool
        If True, the output matrix P is symmetric. This is accomplished 
        by ignoring the order of value pairs, so both (i, j) and (j, i) 
        are accumulated when (i, j) is encountered. 
    normal : bool
        If True, normalize the result by dividing by the number of 
        possible outcomes

    Returns
    -------
    P : 4-dimensional ndarray
       The grey-level co-occurrence histogram. The value
       P[i,j,d,theta] is the number of times that grey-level j
       occurs at a distance d and at an angle theta from
       grey-level i.

    Examples
    --------
    Compute 2 GLCMs: One for a 1-pixel offset to the right, and one
    for a 1-pixel offset upwards.
    
    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> result = glcm(image, [1], [0, np.pi/2], 4)
    >>> result[:, :, 0, 0]
    array([[2, 2, 1, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 1],
           [0, 0, 0, 1]], dtype=uint32)
    >>> result[:, :, 0, 1] 
    array([[3, 0, 2, 0],
           [0, 2, 2, 0],
           [0, 0, 1, 2],
           [0, 0, 0, 0]], dtype=uint32)

    """
    image = skimage.util.img_as_ubyte(image)    
    assert image.ndim == 2
    assert image.min() >= 0
    assert image.max() < levels
    distances = np.asarray(distances)
    angles = np.asarray(angles)
    assert distances.ndim == 1
    assert angles.ndim == 1

    rows, cols = image.shape
    out = np.zeros((levels, levels, len(distances), len(angles)),
                   dtype=np.uint32)
    
    for a_idx, angle in enumerate(angles):
        for d_idx, distance in enumerate(distances):
            for r in range(rows):
                for c in range(cols):
                    i = image[r, c]

                    # compute the location of the offset pixel
                    row = r + int(np.round(np.sin(angle) * distance))
                    col = c + int(np.round(np.cos(angle) * distance))
                    
                    # make sure the offset is within bounds
                    if row >= 0 and row < rows and \
                       col >= 0 and col < cols:
                        j = image[row, col]
                        
                        if i >= 0 and i < levels and \
                           j >= 0 and j < levels:
                            out[i, j, d_idx, a_idx] += 1
                            if symmetric:
                                out[j, i, d_idx, a_idx] += 1

    # normalize
    if normal:
        out = out.astype(np.float64) / out.sum()

    return out

if __name__ == "__main__":
    import doctest
    doctest.testmod()
