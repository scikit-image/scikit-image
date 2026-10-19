import numpy as np
from numpy.testing import assert_equal
import pytest

from _skimage2.morphology import pad_footprint, mirror_footprint


# skimage.morphology.mirror_footprint --------------------------------------------------


@pytest.mark.parametrize("as_sequence", [tuple, None])
def test_mirror_footprint(as_sequence):
    footprint = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], np.uint8)
    expected_res = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=np.uint8)
    if as_sequence is not None:
        footprint = as_sequence([(footprint, 2), (footprint.T, 3)])
        expected_res = as_sequence([(expected_res, 2), (expected_res.T, 3)])

    actual_res = mirror_footprint(footprint)
    assert type(expected_res) is type(actual_res)
    assert_equal(expected_res, actual_res)


# skimage.morphology.pad_footprint -----------------------------------------------------


@pytest.mark.parametrize("as_sequence", [tuple, None])
@pytest.mark.parametrize("pad_end", [True, False])
def test_pad_footprint(as_sequence, pad_end):
    footprint = np.array([[0, 0], [1, 0], [1, 1]], np.uint8)
    pad_width = [(0, 0), (0, 1)] if pad_end is True else [(0, 0), (1, 0)]
    expected_res = np.pad(footprint, pad_width)
    if as_sequence is not None:
        footprint = as_sequence([(footprint, 2), (footprint.T, 3)])
        expected_res = as_sequence([(expected_res, 2), (expected_res.T, 3)])

    actual_res = pad_footprint(footprint, pad_end=pad_end)
    assert type(expected_res) is type(actual_res)
    assert_equal(expected_res, actual_res)
