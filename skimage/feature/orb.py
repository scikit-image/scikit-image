import numpy as np

from ..util import img_as_float
from .util import _mask_border_keypoints

from skimage.feature import (corner_fast, corner_orientations, corner_peaks,
                             corner_harris)
from skimage.transform import pyramid_gaussian


def keypoints_orb(image, n_keypoints=200, fast_n=9, fast_threshold=0.20,
                  harris_k=0.05,  downscale_factor=np.sqrt(2), n_scales=5):

    """Compute Oriented Fast keypoints.

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
    downscale_factor : float
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
    ..[1] Ethan Rublee, Vincent Rabaud, Kurt Konolige and Gary Bradski
          "ORB : An efficient alternative to SIFT and SURF"
          http://www.vision.cs.chubu.ac.jp/CV-R/pdf/Rublee_iccv2011.pdf

    """ 
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Only 2-D gray-scale images supported.")

    pyramid = list(pyramid_gaussian(image, n_scales - 1, downscale_factor))

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
        corners = corner_peaks(corner_fast(pyramid[i], fast_n, fast_threshold), min_distance=1)
        keypoints_list.append(corners)
        orientations_list.append(corner_orientations(pyramid[i], corners, ofast_mask))
        scales_list.append(i * np.ones((corners.shape[0]), dtype=np.intp))
        harris_measure_list.append(harris_response[corners[:, 0], corners[:, 1]])

    keypoints = np.vstack(keypoints_list)
    orientations = np.hstack(orientations_list)
    scales = np.hstack(scales_list)
    harris_measure = np.hstack(harris_measure_list)

    if keypoints.shape[0] < n_keypoints:
        return keypoints, orientations, scales
    else:
        best_indices = harris_measure.argsort()[::-1][:n_keypoints]
        return keypoints[best_indices], orientations[best_indices], scales[best_indices]


def descriptor_orb(image, keypoints, keypoints_angle):

    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Only 2-D gray-scale images supported.")

    image = img_as_float(image)

    mask = _mask_border_keypoints(image, keypoints, 13)
    keypoints = keypoints[mask]
    keypoints_angle = keypoints_angle[mask]

    pos1 = binary_tests[:, :2]
    pos2 = binary_tests[:, 2:]

    descriptors = np.zeros((keypoints.shape[0], 256), dtype=bool)

    for i in range(keypoints.shape[0]):
        angle = keypoints_angle[i]
        a = np.sin(angle * np.pi / 180.)
        b = np.cos(angle * np.pi / 180.)
        rotation_matrix = [[b, a], [-a, b]]
        steered_pos1 = pos1 * rotation_matrix
        steered_pos2 = pos2 * rotation_matrix
        for j in range(256):
            pr1 = steered_pos1[j][0]
            pc1 = steered_pos1[j][1]
            pr2 = steered_pos2[j][0]
            pc2 = steered_pos2[j][1]
            descriptors[i, j] = (image[pr1, pc1] < image[pr2, pc2])
    return descriptors


# Learned 256 decision pairs for binary tests in rBRIEF. Taken from OpenCV.
binary_tests = np.asarray([[8,-3, 9,5],
                           [4,2, 7,-12],
                           [-11,9, -8,2],
                           [7,-12, 12,-13],
                           [2,-13, 2,12],
                           [1,-7, 1,6],
                           [-2,-10, -2,-4],
                           [-13,-13, -11,-8],
                           [-13,-3, -12,-9],
                           [10,4, 11,9],
                           [-13,-8, -8,-9],
                           [-11,7, -9,12],
                           [7,7, 12,6],
                           [-4,-5, -3,0],
                           [-13,2, -12,-3],
                           [-9,0, -7,5],
                           [12,-6, 12,-1],
                           [-3,6, -2,12],
                           [-6,-13, -4,-8],
                           [11,-13, 12,-8],
                           [4,7, 5,1],
                           [5,-3, 10,-3],
                           [3,-7, 6,12],
                           [-8,-7, -6,-2],
                           [-2,11, -1,-10],
                           [-13,12, -8,10],
                           [-7,3, -5,-3],
                           [-4,2, -3,7],
                           [-10,-12, -6,11],
                           [5,-12, 6,-7],
                           [5,-6, 7,-1],
                           [1,0, 4,-5],
                           [9,11, 11,-13],
                           [4,7, 4,12],
                           [2,-1, 4,4],
                           [-4,-12, -2,7],
                           [-8,-5, -7,-10],
                           [4,11, 9,12],
                           [0,-8, 1,-13],
                           [-13,-2, -8,2],
                           [-3,-2, -2,3],
                           [-6,9, -4,-9],
                           [8,12, 10,7],
                           [0,9, 1,3],
                           [7,-5, 11,-10],
                           [-13,-6, -11,0],
                           [10,7, 12,1],
                           [-6,-3, -6,12],
                           [10,-9, 12,-4],
                           [-13,8, -8,-12],
                           [-13,0, -8,-4],
                           [3,3, 7,8],
                           [5,7, 10,-7],
                           [-1,7, 1,-12],
                           [3,-10, 5,6],
                           [2,-4, 3,-10],
                           [-13,0, -13,5],
                           [-13,-7, -12,12],
                           [-13,3, -11,8],
                           [-7,12, -4,7],
                           [6,-10, 12,8],
                           [-9,-1, -7,-6],
                           [-2,-5, 0,12],
                           [-12,5, -7,5],
                           [3,-10, 8,-13],
                           [-7,-7, -4,5],
                           [-3,-2, -1,-7],
                           [2,9, 5,-11],
                           [-11,-13, -5,-13],
                           [-1,6, 0,-1],
                           [5,-3, 5,2],
                           [-4,-13, -4,12],
                           [-9,-6, -9,6],
                           [-12,-10, -8,-4],
                           [10,2, 12,-3],
                           [7,12, 12,12],
                           [-7,-13, -6,5],
                           [-4,9, -3,4],
                           [7,-1, 12,2],
                           [-7,6, -5,1],
                           [-13,11, -12,5],
                           [-3,7, -2,-6],
                           [7,-8, 12,-7],
                           [-13,-7, -11,-12],
                           [1,-3, 12,12],
                           [2,-6, 3,0],
                           [-4,3, -2,-13],
                           [-1,-13, 1,9],
                           [7,1, 8,-6],
                           [1,-1, 3,12],
                           [9,1, 12,6],
                           [-1,-9, -1,3],
                           [-13,-13, -10,5],
                           [7,7, 10,12],
                           [12,-5, 12,9],
                           [6,3, 7,11],
                           [5,-13, 6,10],
                           [2,-12, 2,3],
                           [3,8, 4,-6],
                           [2,6, 12,-13],
                           [9,-12, 10,3],
                           [-8,4, -7,9],
                           [-11,12, -4,-6],
                           [1,12, 2,-8],
                           [6,-9, 7,-4],
                           [2,3, 3,-2],
                           [6,3, 11,0],
                           [3,-3, 8,-8],
                           [7,8, 9,3],
                           [-11,-5, -6,-4],
                           [-10,11, -5,10],
                           [-5,-8, -3,12],
                           [-10,5, -9,0],
                           [8,-1, 12,-6],
                           [4,-6, 6,-11],
                           [-10,12, -8,7],
                           [4,-2, 6,7],
                           [-2,0, -2,12],
                           [-5,-8, -5,2],
                           [7,-6, 10,12],
                           [-9,-13, -8,-8],
                           [-5,-13, -5,-2],
                           [8,-8, 9,-13],
                           [-9,-11, -9,0],
                           [1,-8, 1,-2],
                           [7,-4, 9,1],
                           [-2,1, -1,-4],
                           [11,-6, 12,-11],
                           [-12,-9, -6,4],
                           [3,7, 7,12],
                           [5,5, 10,8],
                           [0,-4, 2,8],
                           [-9,12, -5,-13],
                           [0,7, 2,12],
                           [-1,2, 1,7],
                           [5,11, 7,-9],
                           [3,5, 6,-8],
                           [-13,-4, -8,9],
                           [-5,9, -3,-3],
                           [-4,-7, -3,-12],
                           [6,5, 8,0],
                           [-7,6, -6,12],
                           [-13,6, -5,-2],
                           [1,-10, 3,10],
                           [4,1, 8,-4],
                           [-2,-2, 2,-13],
                           [2,-12, 12,12],
                           [-2,-13, 0,-6],
                           [4,1, 9,3],
                           [-6,-10, -3,-5],
                           [-3,-13, -1,1],
                           [7,5, 12,-11],
                           [4,-2, 5,-7],
                           [-13,9, -9,-5],
                           [7,1, 8,6],
                           [7,-8, 7,6],
                           [-7,-4, -7,1],
                           [-8,11, -7,-8],
                           [-13,6, -12,-8],
                           [2,4, 3,9],
                           [10,-5, 12,3],
                           [-6,-5, -6,7],
                           [8,-3, 9,-8],
                           [2,-12, 2,8],
                           [-11,-2, -10,3],
                           [-12,-13, -7,-9],
                           [-11,0, -10,-5],
                           [5,-3, 11,8],
                           [-2,-13, -1,12],
                           [-1,-8, 0,9],
                           [-13,-11, -12,-5],
                           [-10,-2, -10,11],
                           [-3,9, -2,-13],
                           [2,-3, 3,2],
                           [-9,-13, -4,0],
                           [-4,6, -3,-10],
                           [-4,12, -2,-7],
                           [-6,-11, -4,9],
                           [6,-3, 6,11],
                           [-13,11, -5,5],
                           [11,11, 12,6],
                           [7,-5, 12,-2],
                           [-1,12, 0,7],
                           [-4,-8, -3,-2],
                           [-7,1, -6,7],
                           [-13,-12, -8,-13],
                           [-7,-2, -6,-8],
                           [-8,5, -6,-9],
                           [-5,-1, -4,5],
                           [-13,7, -8,10],
                           [1,5, 5,-13],
                           [1,0, 10,-13],
                           [9,12, 10,-1],
                           [5,-8, 10,-9],
                           [-1,11, 1,-13],
                           [-9,-3, -6,2],
                           [-1,-10, 1,12],
                           [-13,1, -8,-10],
                           [8,-11, 10,-6],
                           [2,-13, 3,-6],
                           [7,-13, 12,-9],
                           [-10,-10, -5,-7],
                           [-10,-8, -8,-13],
                           [4,-6, 8,5],
                           [3,12, 8,-13],
                           [-4,2, -3,-3],
                           [5,-13, 10,-12],
                           [4,-13, 5,-1],
                           [-9,9, -4,3],
                           [0,3, 3,-9],
                           [-12,1, -6,1],
                           [3,2, 4,-8],
                           [-10,-10, -10,9],
                           [8,-13, 12,12],
                           [-8,-12, -6,-5],
                           [2,2, 3,7],
                           [10,6, 11,-8],
                           [6,8, 8,-12],
                           [-7,10, -6,5],
                           [-3,-9, -3,9],
                           [-1,-13, -1,5],
                           [-3,-7, -3,4],
                           [-8,-2, -8,3],
                           [4,2, 12,12],
                           [2,-5, 3,11],
                           [6,-9, 11,-13],
                           [3,-1, 7,12],
                           [11,-1, 12,4],
                           [-3,0, -3,6],
                           [4,-11, 4,12],
                           [2,-4, 2,1],
                           [-10,-6, -8,1],
                           [-13,7, -11,1],
                           [-13,12, -11,-13],
                           [6,0, 11,-13],
                           [0,-1, 1,4],
                           [-13,3, -9,-2],
                           [-9,8, -6,-3],
                           [-13,-6, -8,-2],
                           [5,-9, 8,10],
                           [2,7, 3,-9],
                           [-1,-6, -1,-1],
                           [9,5, 11,-2],
                           [11,-3, 12,-8],
                           [3,0, 3,5],
                           [-1,4, 0,10],
                           [3,-6, 4,5],
                           [-13,0, -10,5],
                           [5,8, 12,11],
                           [8,9, 9,-6],
                           [7,-4, 8,-12],
                           [-10,4, -10,9],
                           [7,3, 12,4],
                           [9,-7, 10,-2],
                           [7,0, 12,-2],
                           [-1,-6, 0,-11]])
