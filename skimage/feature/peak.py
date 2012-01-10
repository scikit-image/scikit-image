import numpy as np
from scipy import ndimage


def peak_min_dist(image, min_distance=10, threshold=0.1):
    """Return coordinates of peaks in an image.

    Candidate peaks are determined by a relative `threshold`, and peaks that
    are too close (as determined by `min_distance`) to larger peaks are
    rejected.

    Parameters
    ----------
    image: ndarray of floats
        Input image.

    min_distance: int, optional
        Minimum number of pixels separating peaks and image boundary.

    threshold: float, optional
        Candidate peaks are calculated as `max(image) * threshold`.

    Returns
    -------
    coordinates : (N, 2) array
        (row, column) coordinates of peaks.
    """
    image = image.copy()
    # Non maximum filter of size 3
    image_max = ndimage.maximum_filter(image, 3, mode='constant')
    mask = (image == image_max)
    image *= mask

    # Remove the image borders
    image[:3] = 0
    image[-3:] = 0
    image[:, :3] = 0
    image[:, -3:] = 0

    # find top corner candidates above a threshold
    corner_threshold = np.max(image.ravel()) * threshold
    image_t = (image >= corner_threshold) * 1

    # get coordinates of candidates
    candidates = image_t.nonzero()
    coords = np.transpose(candidates)

    # ...and their values
    candidate_values = image[candidates]

    # sort candidates
    index = np.argsort(candidate_values)[::-1]

    # store allowed point locations in array
    allowed_locations = np.zeros(image.shape)
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

