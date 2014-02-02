import numpy as np
import os

from skimage.transform import integral_image
from skimage.feature.util import _mask_border_keypoints
from skimage.util import img_as_float
from skimage import data_dir

from _freak_cy import _get_mean_intensity


# Constants
best_pairs = np.loadtxt(os.path.join(data_dir, 'freak_descriptor_pairs.txt'), dtype=np.uint8)
ns = [6, 6, 6, 6, 6, 6, 6, 1]
BIG_RADIUS = 2./3
SMALL_RADIUS = 2./24
unit_space = (BIG_RADIUS - SMALL_RADIUS) / 21
radii = [BIG_RADIUS, BIG_RADIUS - 6 * unit_space, BIG_RADIUS - 11 * unit_space,
         BIG_RADIUS - 15 * unit_space, BIG_RADIUS - 18 * unit_space,
         BIG_RADIUS - 20 * unit_space, SMALL_RADIUS, 0.0]

radii_stretched = []
for radius, n in zip(radii, ns):
    radii_stretched.extend([radius] * n)


def _get_pattern(pattern_scale):
    pattern = []
    for i in range(8):
        for j in range(ns[i]):
            beta = (np.pi / ns[i]) * (i % 2)
            alpha = j * 2 * np.pi / ns[i] + beta 
            pattern.append([radii[i] * np.cos(alpha) * pattern_scale,
                            radii[i] * np.sin(alpha) * pattern_scale])

    pattern = np.asarray(pattern, dtype=np.intp)
    pattern = np.round(pattern)
    return pattern


def _get_freak_orientation(image, keypoint, pattern):
    opos0 = pattern[orientation_pairs[:, 0]] + keypoint
    opos1 = pattern[orientation_pairs[:, 1]] + keypoint
    intensity_diff = image[opos0[:, 0], opos0[:, 1]] - image[opos1[:, 0], opos1[:, 1]]
    directions = opos0 - opos1
    directions_abs = np.sqrt(directions[:, 0] ** 2 + directions[:, 1] ** 2)
    x_dir, y_dir = np.sum(directions * (intensity_diff / directions_abs)[:, None], axis=0)
    return np.arctan2(x_dir, y_dir)


def descriptor_freak(image, keypoints, pattern_scale=22.0):
    """
    Compute FREAK Descriptors for the given set of keypoints from the given
    image.

    Parameters
    ----------
    image : 2D ndarray
        Input grayscale image.
    keypoints : (N, 2) array
        Keypoint locations extracted from ``image``.
    pattern_scale : float
        Scaling or spread of the Decriptor pattern.

    Returns
    -------
    descriptors : (Q, 512) ndarray of dtype bool
        Extracted FREAK decriptors of length 512 bits about Q keypoints
        obtained after filtering the border keypoints.
    keypoints : (Q, 2) ndarray
        Q keypoints obtained after border filtering out the original N
        keypoints.

    References
    ----------
    .. [1] Alexandre Alahi, Raphael Ortiz, Pierre Vandergheynst "FREAK: Fast
           Retina Keypoint",
           http://infoscience.epfl.ch/record/175537/files/2069.pdf

    """
    # Prepare the input image. Use the util function once #669 is merged.
    image = np.squeeze(image)
    if image.ndim != 2:
        raise ValueError("Only 2-D gray-scale images supported.")

    image = img_as_float(image)

    # Remove border keypoints
    keypoints = keypoints[_mask_border_keypoints(image, keypoints, 30)]

    # Generate the FREAK pattern of 43 points
    pattern = _get_pattern(pattern_scale)

    # Orientation of the keypoints.
    # TODO : Cythonize.
    orientations = np.zeros(keypoints.shape[0])
    for i in range(len(keypoints)):
        orientations[i] = _get_freak_orientation(image, keypoints[i], pattern)

    # Integral image for mean intensities about FREAK pattern points
    integral_img = integral_image(image)

    # Initialize descriptor
    descriptors = np.zeros((keypoints.shape[0], 512), dtype=bool)

    # Computing descriptor around each keypoint
    # TODO : Cythonize.
    for i in range(len(keypoints)):
        # Rotate the pattern
        a = np.cos(orientations[i])
        b = np.sin(orientations[i])
        rot_matrix = [[b, a], [a, -b]]
        rotated_pattern = np.dot(pattern, rot_matrix)

        # Mean / Gaussian intensity at 43 points
        pattern_intensities = np.zeros((43))
        _get_mean_intensity(integral_img, keypoints[i][0], keypoints[i][1],
                            rotated_pattern, np.asarray(radii_stretched),
                            pattern_scale, pattern_intensities)
        pos0 = best_pairs[:, 0]
        pos1 = best_pairs[:, 1]
        descriptors[i, :] = pattern_intensities[pos0] < pattern_intensities[pos1]
    return descriptors, keypoints


# 45 pairs for calculating keypoint orientation. Taken from OpenCV.
orientation_pairs = np.array([[ 0,  3],
       [ 1,  4],
       [ 2,  5],
       [ 0,  2],
       [ 1,  3],
       [ 2,  4],
       [ 3,  5],
       [ 4,  0],
       [ 5,  1],
       [ 6,  9],
       [ 7, 10],
       [ 8, 11],
       [ 6,  8],
       [ 7,  9],
       [ 8, 10],
       [ 9, 11],
       [10,  6],
       [11,  7],
       [12, 15],
       [13, 16],
       [14, 17],
       [12, 14],
       [13, 15],
       [14, 16],
       [15, 17],
       [16, 12],
       [17, 13],
       [18, 21],
       [19, 22],
       [20, 23],
       [18, 20],
       [19, 21],
       [20, 22],
       [21, 23],
       [22, 18],
       [23, 19],
       [24, 27],
       [25, 28],
       [26, 29],
       [30, 33],
       [31, 34],
       [32, 35],
       [36, 39],
       [37, 40],
       [38, 41]], dtype=np.uint8)

