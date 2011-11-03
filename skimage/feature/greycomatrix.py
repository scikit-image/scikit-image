"""
Compute grey level co-occurrence matrices (GLCM) and associated 
properties to characterize image textures.
"""

import numpy as np
import skimage.util

from _greycomatrix import _glcm_loop


def compute_glcm(image, distances, angles, levels=256, symmetric=False,
         normed=False):
    """Calculate the grey-level co-occurrence matrix.

    A grey level co-occurence matrix is a histogram of co-occuring
    greyscale values at a given offset over an image.

    Parameters
    ----------
    image : ndarray
        Input image, which is converted to the uint8 data type.
    distances : array_like
        List of pixel pair distance offsets.
    angles : array_like
        List of pixel pair angles in radians.
    levels : int, optional
        The input image should contain integers in [0, levels-1],
        where levels indicate the number of grey-levels counted
        (typically 256 for an 8-bit image). The default is 256.
    symmetric : bool, optional
        If True, the output matrix P is symmetric. This is accomplished 
        by ignoring the order of value pairs, so both (i, j) and (j, i) 
        are accumulated when (i, j) is encountered. The default is False. 
    normed : bool, optional
        If True, normalize the result by dividing by the number of 
        possible outcomes. The elements of the resulting matrix sum
        to 1. The default is False.

    Returns
    -------
    out : ndarray
        The grey-level co-occurrence histogram. The value
        `P[i,j,d,theta]` is the number of times that grey-level `j`
        occurs at a distance `d` and at an angle `theta` from
        grey-level `i`.

    References
    ----------
    .. [1] The GLCM Tutorial Home Page,
           http://www.fp.ucalgary.ca/mhallbey/tutorial.htm
    .. [2] Pattern Recognition Engineering, Morton Nadler & Eric P. 
           Smith
    

    Examples
    --------
    Compute 2 GLCMs: One for a 1-pixel offset to the right, and one
    for a 1-pixel offset upwards.
    
    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> result = compute_glcm(image, [1], [0, np.pi/2], 4)
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
    distances = np.ascontiguousarray(distances, dtype=np.float64)
    angles = np.ascontiguousarray(angles, dtype=np.float64)
    assert distances.ndim == 1
    assert angles.ndim == 1

    rows, cols = image.shape
    out = np.zeros((levels, levels, len(distances), len(angles)),
                   dtype=np.uint32)
    
    # count co-occurances
    _glcm_loop(image, distances, angles, levels, out)

    # make each GLMC symmetric
    if symmetric:
        for d in range(len(distances)):
            for a in range(len(angles)):
                out[:, :, d, a] += out[:, :, d, a].transpose()
                
    # normalize each GLMC
    if normed:
        out = out.astype(np.float64)
        for d in range(len(distances)):
            for a in range(len(angles)):
                if np.any(out[:, :, d, a]):
                    out[:, :, d, a] /= out[:, :, d, a].sum()

    return out


def compute_glcm_prop(P, prop='contrast'):
    """Calculate texture properties of a GLCM.
    
    Compute a feature of a grey level co-occurance matrix to serve as 
    a compact summary of the matrix. The properties are computed as
    follows:

    - 'contrast': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}(i-j)^2`
    - 'dissimilarity': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}\\left|i-j\\right|`
    - 'homogeneity': :math:`\\sum_{i,j=0}^{levels-1}\\frac{P_{i,j}}{1+(i-j)^2}`
    - 'ASM': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}^2`    
    - 'energy': :math:`\\sqrt{ASM}`
    - 'correlation': :math:`\\sum_{i,j=0}^{levels-1} P_{i,j}\\left[\\frac{(i-\\mu_i)(j-\\mu_j)}{\\sqrt{(\\sigma_i^2)(\\sigma_j^2)}}\\right]`

    
    Parameters
    ----------    
    P : ndarray
        Input array. `P` is the grey-level co-occurrence histogram 
        for which to compute the specified property. The value
        `P[i,j,d,theta]` is the number of times that grey-level j
        occurs at a distance d and at an angle theta from
        grey-level i.
    prop : {'contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM'}, optional
        The property of the GLCM to compute. The default is 'contrast'.
    
    Returns
    -------
    results : ndarray
        2-dimensional array. `results[d, a]` is the property 'prop' for 
        the d'th distance and the a'th angle.
    
    References
    ----------
    .. [1] The GLCM Tutorial Home Page,
           http://www.fp.ucalgary.ca/mhallbey/tutorial.htm    
    
    Examples
    --------
    Compute the contrast for GLCMs with distances [1, 2] and angles
    [0 degrees, 90 degrees] 
    
    >>> image = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [0, 2, 2, 2],
    ...                   [2, 2, 3, 3]], dtype=np.uint8)
    >>> g = compute_glcm(image, [1, 2], [0, np.pi/2], 4, normed=True, 
    ...                  symmetric=True)
    >>> contrast = compute_glcm_prop(g, 'contrast')
    >>> contrast
    array([[ 0.58333333,  1.        ],
           [ 1.25      ,  2.75      ]])
    
    """
    
    assert P.ndim == 4
    (num_level, num_level2, num_dist, num_angle) = P.shape
    assert num_level == num_level2
    assert num_dist > 0
    assert num_angle > 0
    
    # create weights for specified property
    r = range(num_level)
    I, J = np.meshgrid(r, r)
    if prop == 'contrast':
        weights = (I - J) ** 2
    elif prop == 'dissimilarity':
        weights = np.abs(I - J)
    elif prop == 'homogeneity':
        weights = 1. / (1. + (I - J) ** 2)
    elif prop in ['ASM', 'energy', 'correlation']:
        pass
    else:
        raise ValueError('%s is an invalid property' % (prop))
    
    # compute property for each GLCM 
    results = np.zeros((num_dist, num_angle), dtype=np.float64)
    for d in range(num_dist):
        for a in range(num_angle):
            if prop == 'energy':
                asm = (P[:, :, d, a] ** 2).sum()
                results[d, a] = np.sqrt(asm)
            elif prop == 'ASM':
                results[d, a] = (P[:, :, d, a] ** 2).sum()
            elif prop == 'correlation':
                g = P[:, :, d, a]
                mean_i = (I * g).sum()
                mean_j = (J * g).sum()
                diff_i = I - mean_i
                diff_j = J - mean_j
                std_i = np.sqrt((g * (diff_i) ** 2).sum())
                std_j = np.sqrt((g * (diff_j) ** 2).sum())
                cov = (g * (diff_i * diff_j)).sum()
                if std_i < 1e-15 or std_j < 1e-15:
                    corr = 1.
                else:
                    corr = cov / (std_i * std_j)
                
                results[d, a] = corr
            else:
                results[d, a] = (P[:, :, d, a] * weights).sum()

    return results
