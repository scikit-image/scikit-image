import numpy as np

from skimage.util import img_as_float


def create_keypoint_recarray(rows, cols, scales=None, orientations=None,
                             responses=None):
    """Create keypoint array that allows field access through attributes.

    Parameters
    ----------
    rows : (N, ) array
        Row coordinates of keypoints.
    cols : (N, ) array
        Column coordinates of keypoints.
    scales : (N, ) array
        Scales in which the keypoints have been detected.
    orientations : (N, ) array
        Orientations of the keypoints.
    responses : (N, ) array
        Detector response (strength) of the keypoints.

    Returns
    -------
    recarray : (N, 5) recarray
        Array with the fields: `row`, `col`, `scale`, `orientation` and
        `response`.

    """

    dtype = [('row', np.double),
             ('col', np.double),
             ('scale', np.double),
             ('orientation', np.double),
             ('response', np.double)]
    keypoints = np.zeros(row.shape[0], dtype=dtype)
    keypoints['row'] = rows
    keypoints['col'] = cols
    keypoints['scale'] = scales
    keypoints['orientation'] = orientations
    keypoints['response'] = responses
    return keypoints.view(np.recarray)


def _prepare_grayscale_input_2D(image):
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Only 2-D gray-scale images supported.")

    return img_as_float(image)


def _mask_border_keypoints(image, keypoints, dist):
    """Removes keypoints that are within dist pixels from the image border."""
    width = image.shape[0]
    height = image.shape[1]

    keypoints_filtering_mask = ((dist - 1 < keypoints[:, 0]) &
                                (keypoints[:, 0] < width - dist + 1) &
                                (dist - 1 < keypoints[:, 1]) &
                                (keypoints[:, 1] < height - dist + 1))

    return keypoints_filtering_mask


def pairwise_hamming_distance(array1, array2):
    """**Experimental function**.

    Calculate hamming dissimilarity measure between two sets of
    vectors.

    Parameters
    ----------
    array1 : (P1, D) array
        P1 vectors of size D.
    array2 : (P2, D) array
        P2 vectors of size D.

    Returns
    -------
    distance : (P1, P2) array of dtype float
        2D ndarray with value at an index (i, j) representing the hamming
        distance in the range [0, 1] between ith vector in array1 and jth
        vector in array2.

    """
    distance = (array1[:, None] != array2[None]).mean(axis=2)
    return distance
