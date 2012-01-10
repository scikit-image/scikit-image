"""
Harris corner detector

Inspired from Solem's implementation
http://www.janeriksolem.net/2009/01/harris-corner-detector-in-python.html
"""

import numpy as np
from scipy import ndimage


def _compute_harris_response(image, eps=1e-6, gaussian_deviation=1):
    """Compute the Harris corner detector response function
    for each pixel in the image

    Parameters
    ----------
    image : ndarray of floats
        Input image.

    eps : float, optional
        Normalisation factor.

    gaussian_deviation : integer, optional
        Standard deviation used for the Gaussian kernel.

    Returns
    --------
    image : (M, N) ndarray
        Harris image response
    """
    if len(image.shape) == 3:
        image = image.mean(axis=2)

    # derivatives
    image = ndimage.gaussian_filter(image, gaussian_deviation)
    imx = ndimage.sobel(image, axis=0, mode='constant')
    imy = ndimage.sobel(image, axis=1, mode='constant')

    Wxx = ndimage.gaussian_filter(imx * imx, 1.5, mode='constant')
    Wxy = ndimage.gaussian_filter(imx * imy, 1.5, mode='constant')
    Wyy = ndimage.gaussian_filter(imy * imy, 1.5, mode='constant')

    # determinant and trace
    Wdet = Wxx * Wyy - Wxy ** 2
    Wtr = Wxx + Wyy
    # Alternate formula for Harris response.
    # Alison Noble, "Descriptions of Image Surfaces", PhD thesis (1989)
    harris = Wdet / (Wtr + eps)

    # Non maximum filter of size 3
    harris_max = ndimage.maximum_filter(harris, 3, mode='constant')
    mask = (harris == harris_max)
    harris *= mask

    # Remove the image borders
    harris[:3] = 0
    harris[-3:] = 0
    harris[:, :3] = 0
    harris[:, -3:] = 0

    return harris


def harris(image, min_distance=10, threshold=0.1, eps=1e-6,
           gaussian_deviation=1):
    """Return corners from a Harris response image

    Parameters
    ----------
    image : ndarray of floats
        Input image.

    min_distance : int, optional
        Minimum number of pixels separating interest points and image boundary.

    threshold : float, optional
        Relative threshold impacting the number of interest points.

    eps : float, optional
        Normalisation factor.

    gaussian_deviation : integer, optional
        Standard deviation used for the Gaussian kernel.

    Returns
    -------
    coordinates : (N, 2) array
        (row, column) coordinates of interest points.
    """
    harrisim = _compute_harris_response(image, eps=eps,
                    gaussian_deviation=gaussian_deviation)

    # find top corner candidates above a threshold
    corner_threshold = np.max(harrisim.ravel()) * threshold
    harrisim_t = (harrisim >= corner_threshold) * 1

    # get coordinates of candidates
    candidates = harrisim_t.nonzero()
    coords = np.transpose(candidates)

    # ...and their values
    candidate_values = harrisim[candidates]

    # sort candidates
    index = np.argsort(candidate_values)

    # store allowed point locations in array
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_distance:-min_distance,
                      min_distance:-min_distance] = 1

    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[tuple(coords[i])] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[
              (coords[i][0] - min_distance):(coords[i][0] + min_distance),
              (coords[i][1] - min_distance):(coords[i][1] + min_distance)] = 0

    return np.array(filtered_coords)

