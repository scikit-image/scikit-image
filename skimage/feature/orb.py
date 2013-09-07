import numpy as np

from .util import _mask_border_keypoints, _prepare_grayscale_input_2D

from skimage.feature import (corner_fast, corner_orientations, corner_peaks,
                             corner_harris)
from skimage.transform import pyramid_gaussian

from .orb_cy import _orb_loop


def keypoints_orb(image, n_keypoints=200, fast_n=9, fast_threshold=0.20,
                  harris_k=0.05,  downscale=np.sqrt(2), n_scales=5):

    """Detect Oriented Fast keypoints.

    Parameters
    ----------
    image : 2D ndarray
        Input grayscale image.
    n_keypoints : int
        Number of keypoints to be returned from this function. The function
        will return best `n_keypoints` if more than n_keypoints are detected
        based on the values of other parameters. If not, then all the detected
        keypoints are returned.
    fast_n : int
        The `n` parameter in `feature.corner_fast`. Minimum number of
        consecutive pixels out of 16 pixels on the circle that should all be
        either brighter or darker w.r.t testpixel. A point c on the circle is
        darker w.r.t test pixel p if `Ic < Ip - threshold` and brighter if
        `Ic > Ip + threshold`. Also stands for the n in `FAST-n` corner
        detector.
    fast_threshold : float
        The `threshold` parameter in `feature.corner_fast`. Threshold used to
        decide whether the pixels on the circle are brighter, darker or
        similar w.r.t. the test pixel. Decrease the threshold when more
        corners are desired and vice-versa.
    harris_k : float
        The `k` parameter in `feature.corner_harris`. Sensitivity factor to
        separate corners from edges, typically in range `[0, 0.2]`. Small
        values of k result in detection of sharp corners.
    downscale : float
        Downscale factor for the image pyramid.
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
    >>> scales
    array([0, 0, 0, 0, 1, 1, 1, 1])

    """
    image = _prepare_grayscale_input_2D(image)

    pyramid = list(pyramid_gaussian(image, n_scales - 1, downscale))

    ofast_mask = np.array([[0, 0, 1, 1, 1, 0, 0],
                           [0, 1, 1, 1, 1, 1, 0],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [1, 1, 1, 1, 1, 1, 1],
                           [0, 1, 1, 1, 1, 1, 0],
                           [0, 0, 1, 1, 1, 0, 0]], dtype=np.uint8)

    keypoints_list = []
    orientations_list = []
    scales_list = []
    harris_measure_list = []

    for i in range(n_scales):
        harris_response = corner_harris(pyramid[i], method='k', k=harris_k)
        corners = corner_peaks(corner_fast(pyramid[i], fast_n, fast_threshold),
                               min_distance=1)
        keypoints_list.append(corners)
        orientations_list.append(corner_orientations(pyramid[i], corners,
                                                     ofast_mask))
        scales_list.append(i * np.ones(corners.shape[0], dtype=np.intp))
        harris_measure_list.append(harris_response[corners[:, 0],
                                   corners[:, 1]])

    keypoints = np.vstack(keypoints_list)
    orientations = np.hstack(orientations_list)
    scales = np.hstack(scales_list)
    harris_measure = np.hstack(harris_measure_list)

    if keypoints.shape[0] < n_keypoints:
        return keypoints, orientations, scales
    else:
        best_indices = harris_measure.argsort()[::-1][:n_keypoints]
        return keypoints[best_indices], orientations[best_indices], \
               scales[best_indices]


def descriptor_orb(image, keypoints, orientations, scales,
                   downscale=np.sqrt(2), n_scales=5):
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
        used in `keypoints_orb`.
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
    >>> keypoints, orientations, scales = keypoints_orb(square, n_keypoints=8, \
                                                        n_scales=2)
    >>> keypoints.shape
    (8, 2)
    >>> descriptors, filtered_keypoints = descriptor_orb(square, keypoints, \
                                                         orientations, scales, \
                                                         n_scales=2)
    >>> filtered_keypoints.shape
    (8, 2)
    >>> descriptors.shape
    (8, 256)

    """
    image = _prepare_grayscale_input_2D(image)

    pyramid = list(pyramid_gaussian(image, n_scales - 1, downscale))

    descriptors_list = []
    filtered_keypoints_list = []
    descriptors = np.empty((0, 256), dtype=np.bool)

    for k in range(n_scales):
        curr_image = np.ascontiguousarray(pyramid[k])

        curr_scale_mask = scales == k
        curr_scale_kpts = keypoints[curr_scale_mask]
        curr_scale_orientation = orientations[curr_scale_mask]

        border_mask = _mask_border_keypoints(curr_image, curr_scale_kpts,
                                             dist=13)
        curr_scale_kpts = curr_scale_kpts[border_mask]
        curr_scale_orientation = curr_scale_orientation[border_mask]

        curr_scale_kpts = np.ascontiguousarray(curr_scale_kpts)
        curr_scale_orientation = np.ascontiguousarray(curr_scale_orientation)
        curr_scale_descriptors = _orb_loop(curr_image, curr_scale_kpts,
                                           curr_scale_orientation)

        descriptors_list.append(curr_scale_descriptors)
        filtered_keypoints_list.append(curr_scale_kpts)

    descriptors = np.vstack(descriptors_list).view(np.bool)
    filtered_keypoints = np.vstack(filtered_keypoints_list)
    return descriptors, filtered_keypoints
