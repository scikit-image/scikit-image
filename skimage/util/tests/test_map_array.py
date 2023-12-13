import numpy as np
import pytest
from skimage.util._map_array import map_array, ArrayMap

from skimage._shared import testing


@pytest.mark.parametrize(
    "dtype",
    [np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32, np.int64],
)
def test_map_array_simple(dtype):
    input_arr = np.array([0, 2, 0, 3, 4, 5, 0], dtype=dtype)
    input_vals = np.array([1, 2, 3, 4, 6], dtype=dtype)[::-1]
    output_vals = np.array([6, 7, 8, 9, 10], dtype=dtype)[::-1]
    desired = np.array([0, 7, 0, 8, 9, 0, 0], dtype=dtype)
    result = map_array(
        input_arr=input_arr, input_vals=input_vals, output_vals=output_vals
    )
    np.testing.assert_array_equal(result, desired)
    assert result.dtype == dtype


def test_map_array_incorrect_output_shape():
    labels = np.random.randint(0, 5, size=(24, 25))
    out = np.empty((24, 24))
    in_values = np.unique(labels)
    out_values = np.random.random(in_values.shape).astype(out.dtype)
    with testing.raises(ValueError):
        map_array(labels, in_values, out_values, out=out)


def test_map_array_non_contiguous_output_array():
    labels = np.random.randint(0, 5, size=(24, 25))
    out = np.empty((24 * 3, 25 * 2))[::3, ::2]
    in_values = np.unique(labels)
    out_values = np.random.random(in_values.shape).astype(out.dtype)
    with testing.raises(ValueError):
        map_array(labels, in_values, out_values, out=out)


def test_arraymap_long_str():
    labels = np.random.randint(0, 40, size=(24, 25))
    in_values = np.unique(labels)
    out_values = np.random.random(in_values.shape)
    m = ArrayMap(in_values, out_values)
    assert len(str(m).split('\n')) == m._max_str_lines + 2


def test_arraymap_update():
    in_values = np.unique(np.random.randint(0, 200, size=5))
    out_values = np.random.random(len(in_values))
    m = ArrayMap(in_values, out_values)
    image = np.random.randint(1, len(m), size=(512, 512))
    assert np.all(m[image] < 1)  # missing values map to 0.
    m[1:] += 1
    assert np.all(m[image] >= 1)


def test_arraymap_bool_index():
    in_values = np.unique(np.random.randint(0, 200, size=5))
    out_values = np.random.random(len(in_values))
    m = ArrayMap(in_values, out_values)
    image = np.random.randint(1, len(in_values), size=(512, 512))
    assert np.all(m[image] < 1)  # missing values map to 0.
    positive = np.ones(len(m), dtype=bool)
    positive[0] = False
    m[positive] += 1
    assert np.all(m[image] >= 1)
