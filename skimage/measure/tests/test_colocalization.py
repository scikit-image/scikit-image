import numpy as np
import pytest
from skimage.measure import (pcc, mcc, moc, intersection_coefficient,
                             pixel_intensity_sum, av_pixel_intensity)


def test_invalid_input():
    # images are not same size
    img1 = np.array([[i + j for j in range(4)] for i in range(4)])
    img2 = np.ones((3, 5, 6))
    mask = np.array([[i <= 1 for j in range(5)] for i in range(5)])
    non_binary_mask = np.array([[2 for j in range(4)] for i in range(4)])

    with pytest.raises(ValueError):
        pcc(img1, img1, mask)
    with pytest.raises(ValueError):
        pcc(img1, img2)
    with pytest.raises(ValueError):
        pcc(img1, img1, mask)
    with pytest.raises(ValueError):
        pcc(img1, img1, non_binary_mask)
    with pytest.raises(ValueError):
        mcc(img1, mask)
    with pytest.raises(ValueError):
        mcc(img1, non_binary_mask)
    with pytest.raises(ValueError):
        mcc(img1, img1 > 0, mask)
    with pytest.raises(ValueError):
        mcc(img1, img1 > 0, non_binary_mask)
    with pytest.raises(ValueError):
        moc(img1, img1, mask)
    with pytest.raises(ValueError):
        moc(img1, img2)
    with pytest.raises(ValueError):
        moc(img1, img1, mask)
    with pytest.raises(ValueError):
        moc(img1, img1, non_binary_mask)
    with pytest.raises(ValueError):
        intersection_coefficient(img1 > 2, img2 > 1, mask)
    with pytest.raises(ValueError):
        intersection_coefficient(img1, img2)
    with pytest.raises(ValueError):
        intersection_coefficient(img1, img1, mask)
    with pytest.raises(ValueError):
        intersection_coefficient(img1 > 2, img1 > 1, non_binary_mask)
    with pytest.raises(ValueError):
        pixel_intensity_sum(img1, mask)
    with pytest.raises(ValueError):
        pixel_intensity_sum(img1, non_binary_mask)
    with pytest.raises(ValueError):
        av_pixel_intensity(img1, mask)
    with pytest.raises(ValueError):
        av_pixel_intensity(img1, non_binary_mask)


def test_pcc():
    # simple example
    img1 = np.array([[i + j for j in range(4)] for i in range(4)])
    assert pcc(img1, img1) == (1.0, 0.0)

    img2 = np.where(img1 <= 2, 0, img1)
    assert pcc(img1, img2) == (0.944911182523068, 3.5667540654536515e-08)

    # change background of roi and see if values are same
    roi = np.where(img1 <= 2, 0, 1)
    assert pcc(img1, img1, roi=roi) == pcc(img1, img2, roi=roi)


def test_mcc():
    img1 = np.array([[j for j in range(4)] for i in range(4)])
    mask = np.array([[i <= 1 for j in range(4)]for i in range(4)])
    assert mcc(img1, imgB_mask=mask) == 0.5


def test_moc():
    img1 = np.ones((4, 4))
    img2 = 2*np.ones((4, 4))
    assert moc(img1, img2) == 1


def test_intersection_coefficient():
    img1_mask = np.array([[j <= 1 for j in range(4)] for i in range(4)])
    img2_mask = np.array([[i <= 1 for j in range(4)] for i in range(4)])
    img3_mask = np.array([[1 for j in range(4)] for i in range(4)])
    assert intersection_coefficient(img1_mask, img2_mask) == 0.5
    assert intersection_coefficient(img1_mask, img3_mask) == 1


def test_pixel_intensity_sum():
    img1 = np.array([[i + j for j in range(4)] for i in range(4)])
    assert pixel_intensity_sum(img1) == 48


def test_av_pixel_intensity():
    img1 = np.array([[i + j for j in range(4)] for i in range(4)])
    assert av_pixel_intensity(img1) == 3
