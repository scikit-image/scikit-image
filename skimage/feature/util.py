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
    recarray : (N, ...) recarray
        Array with the fields: `row`, `col`, `scale`, `orientation` and
        `response`.

    """

    dtype = [('row', np.double),
             ('col', np.double),
             ('scale', np.double),
             ('orientation', np.double),
             ('response', np.double)]
    keypoints = np.zeros(rows.shape[0], dtype=dtype)
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


def _mask_border_keypoints(shape, rr, cc, distance):
    """Mask coordinates that are within certain distance from the image border.

    Parameters
    ----------
    shape : (2, ) array_like
        Shape of the image as ``(rows, cols)``.
    rr : (N, ) array
        Row coordinates.
    cc : (N, ) array
        Column coordinates.
    distance : int
        Image border distance.

    Returns
    -------
    mask : (N, ) bool array
        Mask indicating if pixels are within the image (``True``) or in the
        border region of the image (``False``).

    """

    rows = shape[0]
    cols = shape[1]

    mask = (((distance - 1) < rr)
            & (rr < (rows - distance + 1))
            & ((distance - 1) < cc)
            & (cc < (cols - distance + 1)))

    return mask


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
