import numpy as np
import pytest

from skimage.data import astronaut, cat
from skimage.registration import find_transform_ecc
from skimage.transform import AffineTransform, warp


def test_find_transform_ecc_translation():
    ir = cat()
    translation = AffineTransform(translation=(25, 15))
    iw = warp(ir, translation.inverse)
    warp_matrix = find_transform_ecc(ir, iw, motion_type="translation")
    expected_matrix = translation.params
    np.testing.assert_almost_equal(warp_matrix, expected_matrix, decimal=1)


def test_find_transform_ecc_affine():
    ir = cat()
    affine = AffineTransform(
        scale=(1.5, 1.2), rotation=np.deg2rad(30), translation=(20, 10)
    )
    iw = warp(ir, affine.inverse)
    warp_matrix = find_transform_ecc(ir, iw, motion_type="affine")
    expected_matrix = affine.params
    np.testing.assert_almost_equal(warp_matrix, expected_matrix, decimal=1)


def test_find_transform_ecc_euclidean():
    ir = cat()
    affine = AffineTransform(rotation=np.deg2rad(90))
    iw = warp(ir, affine.inverse)
    warp_matrix = find_transform_ecc(ir, iw, motion_type="affine")
    expected_matrix = affine.params
    np.testing.assert_almost_equal(warp_matrix, expected_matrix, decimal=1)


def test_find_transform_ecc_homography():
    ir = cat()
    iw = ir
    warp_matrix = find_transform_ecc(ir, iw, motion_type="homography")
    expected_matrix = np.eye(3)
    np.testing.assert_almost_equal(warp_matrix, expected_matrix, decimal=1)


def test_find_transform_ecc_uncorrelated():
    ir = cat()
    iw = astronaut()
    with pytest.raises(
        ValueError,
        match="The algorithm stopped before its convergence. The correlation is going to be minimized. Images may be uncorrelated or non-overlapped.",
    ):
        find_transform_ecc(ir, iw, motion_type="affine")
