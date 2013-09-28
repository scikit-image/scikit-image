import numpy as np

from skimage.feature.util import (_mask_border_keypoints,
                                  _prepare_grayscale_input_2D)

from skimage.feature import (corner_fast, corner_orientations, corner_peaks,
                             corner_harris)
from skimage.transform import pyramid_gaussian

from .orb_cy import _orb_loop


OFAST_MASK = np.zeros((31, 31))
umax = [15, 15, 15, 15, 14, 14, 14, 13, 13, 12, 11, 10, 9, 8, 6, 3]
for i in range(-15, 16):
    for j in range(-umax[np.abs(i)], umax[np.abs(i)] + 1):
        OFAST_MASK[15 + j, 15 + i]  = 1


def keypoints_orb(image, n_keypoints=500, fast_n=9, fast_threshold=0.08,
                  harris_k=0.04,  downscale=1.2, n_scales=8):

    """Detect Oriented Fast keypoints.

    Parameters
    ----------
    image : 2D ndarray
        Input grayscale image.
    n_keypoints : int
        Number of keypoints to be returned from this function. The function
        will return best ``n_keypoints`` if more than n_keypoints are detected
        based on the values of other parameters. If not, then all the detected
        keypoints are returned.
    fast_n : int
        The ``n`` parameter in ``feature.corner_fast``. Minimum number of
        consecutive pixels out of 16 pixels on the circle that should all be
        either brighter or darker w.r.t testpixel. A point c on the circle is
        darker w.r.t test pixel p if ``Ic < Ip - threshold`` and brighter if
        ``Ic > Ip + threshold``. Also stands for the n in ``FAST-n`` corner
        detector.
    fast_threshold : float
        The ``threshold`` parameter in ``feature.corner_fast``. Threshold used to
        decide whether the pixels on the circle are brighter, darker or
        similar w.r.t. the test pixel. Decrease the threshold when more
        corners are desired and vice-versa.
    harris_k : float
        The ``k`` parameter in ``feature.corner_harris``. Sensitivity factor to
        separate corners from edges, typically in range ``[0, 0.2]``. Small
        values of k result in detection of sharp corners.
    downscale : float
        Downscale factor for the image pyramid. Default value 1.2 is chosen so
        that we have more scales per octave i.e. log1.2(2) in this case.
    n_scales : int
        Number of scales from the bottom of the image pyramid to extract
        the features from.

    Returns
    -------
    keypoints : (N, 2) ndarray
        The oFAST keypoints.
    orientations : (N,) ndarray
        The orientations of the N extracted keypoints.
    scales : (N,) ndarray
        The scales of the N extracted keypoints.

    References
    ----------
    .. [1] Ethan Rublee, Vincent Rabaud, Kurt Konolige and Gary Bradski
          "ORB : An efficient alternative to SIFT and SURF"
          http://www.vision.cs.chubu.ac.jp/CV-R/pdf/Rublee_iccv2011.pdf

    Examples
    --------
    >>> from skimage.feature import keypoints_orb, descriptor_orb
    >>> square = np.zeros((50, 50))
    >>> square[20:30, 20:30] = 1
    >>> keypoints, orientations, scales = keypoints_orb(square, n_keypoints=8, n_scales=2)
    >>> keypoints.shape
    (8, 2)
    >>> keypoints
    array([[29, 29],
           [29, 20],
           [20, 29],
           [20, 20],
           [15, 15],
           [15, 20],
           [20, 15],
           [20, 20]])
    >>> orientations
    array([-2.35619449, -0.78539816,  2.35619449,  0.78539816,  0.78539816,
           2.35619449, -0.78539816, -2.35619449])
    >>> np.rad2deg(orientations)
    array([-135.,  -45.,  135.,   45.,   45.,  135.,  -45., -135.])
    >>> scales
    array([0, 0, 0, 0, 1, 1, 1, 1])

    """

    image = _prepare_grayscale_input_2D(image)

    pyramid = list(pyramid_gaussian(image, n_scales - 1, downscale))

    keypoints_list = []
    orientations_list = []
    scales_list = []
    harris_response_list = []

    for scale in range(n_scales):

        corners = corner_peaks(corner_fast(pyramid[scale], fast_n,
                                           fast_threshold), min_distance=1)
        keypoints_list.append(corners * downscale ** scale)

        orientations_list.append(corner_orientations(pyramid[scale], corners,
                                                     OFAST_MASK))

        scales_list.append(scale * np.ones(corners.shape[0], dtype=np.intp))

        harris_response = corner_harris(pyramid[scale], method='k', k=harris_k)
        harris_response_list.append(harris_response[corners[:, 0],
                                                    corners[:, 1]])

    keypoints = np.round(np.vstack(keypoints_list)).astype(np.intp)
    orientations = np.hstack(orientations_list)
    scales = np.hstack(scales_list)
    harris_measure = np.hstack(harris_response_list)

    if keypoints.shape[0] < n_keypoints:
        return keypoints, orientations, scales
    else:
        best_indices = harris_measure.argsort()[::-1][:n_keypoints]
        return (keypoints[best_indices], orientations[best_indices],
                scales[best_indices])


def descriptor_orb(image, keypoints, orientations, scales,
                   downscale=1.2, n_scales=8):
    """Compute rBRIEF descriptors of input keypoints.

    Parameters
    ----------
    image : 2D ndarray
        Input grayscale image.
    keypoints : (N, 2) ndarray
        Array of N input keypoint locations in the format (row, col).
    orientations : (N,) ndarray
        The orientations of the corresponding N keypoints.
    scales : (N,) ndarray
        The scales of the corresponding N keypoints.
    downscale : float
        Downscale factor for the image pyramid. Should be the same as that
        used in ``keypoints_orb``.
    n_scales : int
        Number of scales from the bottom of the image pyramid to extract
        the features from.

    Returns
    -------
    descriptors : (P, 256) bool ndarray
        2darray of type bool describing the P keypoints obtained after
        filtering out those near the image border. Size of each descriptor
        is 32 bytes or 256 bits.
    filtered_keypoints : (P, 2) ndarray
        Location i.e. (row, col) of P keypoints after removing out those that
        are near border.

    References
    ----------
    .. [1] Ethan Rublee, Vincent Rabaud, Kurt Konolige and Gary Bradski
          "ORB : An efficient alternative to SIFT and SURF"
          http://www.vision.cs.chubu.ac.jp/CV-R/pdf/Rublee_iccv2011.pdf

    Examples
    --------
    >>> import numpy as np
    >>> from skimage.feature import keypoints_orb, descriptor_orb
    >>> square = np.zeros((50, 50))
    >>> square[20:30, 20:30] = 1
    >>> keypoints, orientations, scales = keypoints_orb(square, n_keypoints=8,
        ... n_scales=2)
    >>> keypoints.shape
    (8, 2)
    >>> descriptors, filtered_keypoints = descriptor_orb(square, keypoints,
        ... orientations, scales,
        ... n_scales=2)
    >>> filtered_keypoints.shape
    (8, 2)
    >>> descriptors.shape
    (8, 256)

    """
    image = _prepare_grayscale_input_2D(image)

    pyramid = list(pyramid_gaussian(image, n_scales - 1, downscale))

    descriptors_list = []
    filtered_keypoints_list = []

    for scale in range(n_scales):
        curr_image = np.ascontiguousarray(pyramid[scale])

        curr_scale_mask = scales == scale
        curr_scale_kpts = keypoints[curr_scale_mask] / (downscale ** scale)
        curr_scale_kpts = np.round(curr_scale_kpts).astype(np.intp)
        curr_scale_orientation = orientations[curr_scale_mask]

        border_mask = _mask_border_keypoints(curr_image, curr_scale_kpts,
                                             dist=16)
        curr_scale_kpts = curr_scale_kpts[border_mask]
        curr_scale_orientation = curr_scale_orientation[border_mask]

        curr_scale_kpts = np.ascontiguousarray(curr_scale_kpts)
        curr_scale_orientation = np.ascontiguousarray(curr_scale_orientation)
        curr_scale_descriptors = _orb_loop(curr_image, curr_scale_kpts,
                                           curr_scale_orientation)

        descriptors_list.append(curr_scale_descriptors)
        filtered_keypoints_list.append(curr_scale_kpts * downscale ** scale)

    descriptors = np.vstack(descriptors_list).view(np.bool)
    filtered_keypoints = np.vstack(filtered_keypoints_list)
    filtered_keypoints = np.round(filtered_keypoints).astype(np.intp)
    return descriptors, filtered_keypoints
