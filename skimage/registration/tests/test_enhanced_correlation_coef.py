import numpy as np
import scipy.ndimage as ndi

from skimage.color import rgb2gray
from skimage.data import astronaut, cells3d
from skimage.registration import custom_warp, find_transform_ecc
from skimage.transform import AffineTransform

# Taken from PR #7421, to be replace once it is merged.
max_error = 4


def target_registration_error(shape, matrix):
    """
    Compute the displacement norm of the transform at at each pixel.
    Parameters
    ----------
    shape : shape like
        Shape of the array.
    matrix: ndarray
        Homogeneous matrix.
    Returns
    -------
    error : ndarray
        Norm of the displacement given by the transform.
    """
    # Create a regular set of points on the grid
    slc = [slice(0, n) for n in shape]
    N = np.prod(shape)
    points = np.stack([*[x.ravel() for x in np.mgrid[slc]], np.ones(N)])
    # compute the displacement
    delta = matrix @ points - points
    error = np.linalg.norm(delta[: len(shape)], axis=0).reshape(shape)
    return error


# Based on the implementation of tests from PR #7421
def test_find_transform_ecc_translation():
    ir = rgb2gray(astronaut())[::2, ::2]
    forward = AffineTransform(translation=(15, -20))
    iw = ndi.affine_transform(ir, forward, order=1)
    mat = find_transform_ecc(ir, iw, motion_type='translation', termination_eps=1e-12)
    tre = target_registration_error(ir.shape, mat @ forward)
    assert (
        tre.max() < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."


def test_find_transform_ecc_euclidean():
    ir = rgb2gray(astronaut())[::2, ::2]
    forward = AffineTransform(rotation=0.15)
    iw = ndi.affine_transform(ir, forward, order=1)
    mat = find_transform_ecc(ir, iw, motion_type='euclidean', termination_eps=1e-12)
    tre = target_registration_error(ir.shape, mat @ forward)
    assert (
        tre.max() < max_error
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."


def test_find_transform_ecc_affine():
    ir = rgb2gray(astronaut())[::2, ::2]
    forward = AffineTransform(translation=(15, -20), rotation=0.15, shear=0.15)
    iw = ndi.affine_transform(ir, forward, order=1)
    mat = find_transform_ecc(ir, iw, motion_type='affine', termination_eps=1e-12)
    tre = target_registration_error(ir.shape, mat @ forward)
    assert (
        tre.max()
        < 2
        * max_error  # Correction appears to be a bit less precise with an homography
        # Probably also due to the ir is downsampled by 4 for speed reasons.
    ), f"TRE ({tre.max():.2f}) is more than {2 * max_error} pixels."


# This test might be too long.
def test_find_transform_ecc_homography():
    ir = rgb2gray(astronaut())[::4, ::4]
    forward = AffineTransform(translation=(10, -15), rotation=0.14, shear=0.01).params
    forward[2, 0] = 1e-3
    iw = custom_warp(ir, forward, motion_type='homography')
    mat = find_transform_ecc(ir, iw, motion_type='homography', termination_eps=1e-12)
    tre = target_registration_error(ir.shape, mat @ forward)
    assert (
        tre.max()
        < 2
        * max_error  # Correction appears to be a bit less precise with an homography
        # Probably also due to the ir is downsampled by 4 for speed reasons.
    ), f"TRE ({tre.max():.2f}) is more than {2 * max_error} pixels."


def test_find_transform_ecc_translation_3D():
    ir = cells3d()[:, 0, :, :]
    forward = np.identity(4)
    forward[0, 3] = 3
    forward[1, 3] = 7
    forward[2, 3] = -5
    iw = ndi.affine_transform(ir, forward, order=1)
    mat = find_transform_ecc(ir, iw, motion_type='translation', termination_eps=1e-6)
    tre = target_registration_error(ir.shape, mat @ forward)
    assert (
        tre.max()
        < max_error  # Correction appears to be a bit less precise with an homography
        # Probably also due to the ir is downsampled by 4 for speed reasons.
    ), f"TRE ({tre.max():.2f}) is more than {max_error} pixels."
