import pytest

import numpy as np

from skimage.registration._lddmm_utilities import _validate_scalar_to_multi
from skimage.registration._lddmm_utilities import _validate_ndarray
from skimage.registration._lddmm_utilities import _validate_resolution
from skimage.registration._lddmm_utilities import _compute_axes
from skimage.registration._lddmm_utilities import _compute_coords
from skimage.registration._lddmm_utilities import _multiply_coords_by_affine
from skimage.registration._lddmm_utilities import resample

"""
Test _validate_scalar_to_multi.
"""

def test__validate_scalar_to_multi():

    # Test proper use.

    kwargs = dict(value=1, size=1, dtype=float)
    correct_output = np.array([1], float)
    assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    kwargs = dict(value=1, size=0, dtype=int)
    correct_output = np.array([], int)
    assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    kwargs = dict(value=9.5, size=4, dtype=int)
    correct_output = np.full(4, 9, int)
    assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    kwargs = dict(value=[1, 2, 3.5], size=3, dtype=float)
    correct_output = np.array([1, 2, 3.5], float)
    assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    kwargs = dict(value=[1, 2, 3.5], size=3, dtype=int)
    correct_output = np.array([1, 2, 3], int)
    assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    kwargs = dict(value=(1, 2, 3), size=3, dtype=int)
    correct_output = np.array([1, 2, 3], int)
    assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    kwargs = dict(value=np.array([1, 2, 3], float), size=3, dtype=int)
    correct_output = np.array([1, 2, 3], int)
    assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    kwargs = dict(value=np.array([1, 2, 3], float), size=None, dtype=float)
    correct_output = np.array([1, 2, 3], float)
    assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)


    kwargs = dict(value=1, size=None, dtype=float)
    correct_output = np.array([1], float)
    assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    # Test improper use.

    kwargs = dict(value=[1, 2, 3, 4], size='size: not an int', dtype=float)
    expected_exception = TypeError
    match = "size must be either None or interpretable as an integer."
    with pytest.raises(expected_exception, match=match):
        _validate_scalar_to_multi(**kwargs)

    kwargs = dict(value=[], size=-1, dtype=float)
    expected_exception = ValueError
    match = "size must be non-negative."
    with pytest.raises(expected_exception, match=match):
        _validate_scalar_to_multi(**kwargs)

    kwargs = dict(value=[1, 2, 3, 4], size=3, dtype=int)
    expected_exception = ValueError
    match = "The length of value must either be 1 or it must match size if size is provided."
    with pytest.raises(expected_exception, match=match):
        _validate_scalar_to_multi(**kwargs)

    kwargs = dict(value=np.arange(3*4, dtype=int).reshape(3,4), size=3, dtype=float)
    expected_exception = ValueError
    match = "value must not have more than 1 dimension."
    with pytest.raises(expected_exception, match=match):
        _validate_scalar_to_multi(**kwargs)

    kwargs = dict(value=[1, 2, 'c'], size=3, dtype=int)
    expected_exception = ValueError
    match = "value and dtype are incompatible with one another."
    with pytest.raises(expected_exception, match=match):
        _validate_scalar_to_multi(**kwargs)

    kwargs = dict(value='c', size=3, dtype=int)
    expected_exception = ValueError
    match = "value and dtype are incompatible with one another."
    with pytest.raises(expected_exception, match=match):
        _validate_scalar_to_multi(**kwargs)

"""
Test _validate_ndarray.
"""

def test__validate_ndarray():

    # Test proper use.

    kwargs = dict(array=np.arange(3, dtype=int), dtype=float)
    correct_output = np.arange(3, dtype=float)
    assert np.array_equal(_validate_ndarray(**kwargs), correct_output)

    kwargs = dict(array=[[0,1,2], [3,4,5]], dtype=float)
    correct_output = np.arange(2*3, dtype=float).reshape(2,3)
    assert np.array_equal(_validate_ndarray(**kwargs), correct_output)

    kwargs = dict(array=np.array(7), required_ndim=1)
    correct_output = np.array([7])
    assert np.array_equal(_validate_ndarray(**kwargs), correct_output)

    kwargs = dict(array=np.array([0,1,2]), broadcast_to_shape=(2,3))
    correct_output = np.array([[0,1,2], [0,1,2]])
    assert np.array_equal(_validate_ndarray(**kwargs), correct_output)

    kwargs = dict(array=np.arange(3 * 4).reshape(3, 4), reshape_to_shape=(4, 3))
    correct_output = np.arange(3 * 4).reshape(4, 3)
    assert np.array_equal(_validate_ndarray(**kwargs), correct_output)

    kwargs = dict(array=np.arange(3 * 4).reshape(3, 4), required_shape=(3, -1))
    correct_output = np.araange(3 * 4).reshape(3, 4)
    assert np.array_equal(_validate_ndarray(**kwargs), correct_output)
    # Test improper use.

    # Validate arguments.

    kwargs = dict(array=np.arange(3), minimum_ndim=1.5)
    expected_exception = TypeError
    match = "minimum_ndim must be of type int."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

    kwargs = dict(array=np.arange(3), minimum_ndim=-1)
    expected_exception = ValueError
    match = "minimum_ndim must be non-negative."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

    kwargs = dict(array=np.arange(3), required_ndim=1.5)
    expected_exception = TypeError
    match = "required_ndim must be either None or of type int."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

    kwargs = dict(array=np.arange(3), required_ndim=-1)
    expected_exception = ValueError
    match = "required_ndim must be non-negative."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

    kwargs = dict(array=np.arange(3), dtype="not of type type")
    expected_exception = TypeError
    match = "dtype must be either None or a valid type."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)
    
    # Validate array.

    kwargs = dict(array=np.array(print), dtype=int)
    expected_exception = TypeError
    match = "array is of a type that is incompatible with dtype."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

    kwargs = dict(array=np.array('string that is not an int'), dtype=int)
    expected_exception = ValueError
    match = "array has a value that is incompatible with dtype."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)
    
    kwargs = dict(array=np.array([[], 1]), dtype=None, forbid_object_dtype=True)
    expected_exception = TypeError
    match = "Casting array to a np.ndarray produces an array of dtype object \nwhile forbid_object_dtype == True and dtype != object."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

    kwargs = dict(array=np.arange(3), required_ndim=2)
    expected_exception = ValueError
    match = "If required_ndim is not None, array.ndim must equal it unless array.ndim == 0 and required_ndin == 1."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

    kwargs = dict(array=np.arange(3), minimum_ndim=2)
    expected_exception = ValueError
    match = "array.ndim must be at least equal to minimum_ndim."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

    kwargs = dict(array=np.arange(3 * 4).reshape(3, 4), required_shape=(3, 5))
    expected_exception = ValueError
    match = "array must match required_shape."
    with pytest.raises(expected_exception, match=match):
        _validate_ndarray(**kwargs)

"""
Test _validate_resolution.
"""

def test__validate_resolution():

    # Test proper use.
    
    kwargs = dict(resolution=2, ndim=1)
    correct_output = np.full(1, 2, float)
    assert np.array_equal(_validate_resolution(**kwargs), correct_output)

    kwargs = dict(resolution=1.5, ndim=4)
    correct_output = np.full(4, 1.5, float)
    assert np.array_equal(_validate_resolution(**kwargs), correct_output)

    kwargs = dict(resolution=np.ones(3, int), ndim=3)
    correct_output = np.ones(3, float)
    assert np.array_equal(_validate_resolution(**kwargs), correct_output)

    kwargs = dict(resolution=[3, 4], ndim=2)
    correct_output = np.array([3, 4], float)
    assert np.array_equal(_validate_resolution(**kwargs), correct_output)

    # Test improper use.

    kwargs = dict(resolution=[3, -4], ndim=2)
    expected_exception = ValueError
    match = "All elements of resolution must be positive."
    with pytest.raises(expected_exception, match=match):
        _validate_resolution(**kwargs)

    kwargs = dict(resolution=[3, 0], ndim=2)
    expected_exception = ValueError
    match = "All elements of resolution must be positive."
    with pytest.raises(expected_exception, match=match):
        _validate_resolution(**kwargs)

"""
Test _compute_axes.
"""

def test__compute_axes():

    # Test proper use.

    # _compute_axes produces a list with a np.ndarray for each element in shape.

    kwargs = dict(shape=(0, 1, 2), resolution=1, origin='center')
    correct_output = [np.arange(dim_size) * dim_res - np.mean(np.arange(dim_size) * dim_res) 
        for dim_size, dim_res in zip((0, 1, 2), (1, 1, 1))]
    for dim, coord in enumerate(_compute_axes(**kwargs)):
        assert np.array_equal(coord, correct_output[dim])

    kwargs = dict(shape=(1, 2, 3, 4), resolution=1.5, origin='center')
    correct_output = [np.arange(dim_size) * dim_res - np.mean(np.arange(dim_size) * dim_res) 
        for dim_size, dim_res in zip((1, 2, 3, 4), (1.5, 1.5, 1.5, 1.5))]
    for dim, coord in enumerate(_compute_axes(**kwargs)):
        assert np.array_equal(coord, correct_output[dim])

    kwargs = dict(shape=(2, 3, 4), resolution=[1, 1.5, 2], origin='center')
    correct_output = [np.arange(dim_size) * dim_res - np.mean(np.arange(dim_size) * dim_res) 
        for dim_size, dim_res in zip((2, 3, 4), (1, 1.5, 2))]
    for dim, coord in enumerate(_compute_axes(**kwargs)):
        assert np.array_equal(coord, correct_output[dim])

    kwargs = dict(shape=5, resolution=1, origin='center')
    correct_output = [np.arange(dim_size) * dim_res - np.mean(np.arange(dim_size) * dim_res) 
        for dim_size, dim_res in zip((5,), (1,))]
    for dim, coord in enumerate(_compute_axes(**kwargs)):
        assert np.array_equal(coord, correct_output[dim])

    kwargs = dict(shape=5, resolution=1, origin='zero')
    correct_output = [np.arange(dim_size) * dim_res
        for dim_size, dim_res in zip((5,), (1,))]
    for dim, coord in enumerate(_compute_axes(**kwargs)):
        assert np.array_equal(coord, correct_output[dim])

"""
Test _compute_coords.
"""

def test__compute_coords():

    # Test proper use.

    kwargs = dict(shape=5, resolution=1, origin='center')
    correct_output = np.array([[-2], [-1], [0], [1], [2]])
    assert np.array_equal(_compute_coords(**kwargs), correct_output)

    kwargs = dict(shape=(3,4), resolution=1, origin='zero')
    correct_output = np.array([[[0,0], [0,1], [0,2], [0,3]], [[1,0], [1,1], [1,2], [1,3]], [[2,0], [2,1], [2,2], [2,3]]])
    assert np.array_equal(_compute_coords(**kwargs), correct_output)

"""
Test _multiply_coords_by_affine.
"""

def test__multiply_coords_by_affine():

    # Test proper use.

    # Test 3D case.

    array = _compute_coords((3,4,5), 1)
    affine = np.eye(4) + np.append(np.arange(3*4).reshape(3,4), np.zeros((1,4)), 0)**2
    result = _multiply_coords_by_affine(affine, array)
    arrays = []
    for dim in range(3):
        arrays.append(np.sum(affine[dim, :-1] * array, axis=-1) + affine[dim, -1])
    expected = np.stack(arrays=arrays, axis=-1)

    assert np.array_equal(result, expected)

    # Test improper use.

    # Verify affine is 2-dimensional.
    affine = np.eye(4)[None]
    array = np.arange(4*5*6*3).reshape(4,5,6,3)
    expected_exception = ValueError
    match = "affine must be a 2-dimensional matrix."
    with pytest.raises(expected_exception, match=match):
        _multiply_coords_by_affine(affine, array)

    # Verify affine is square.
    affine = np.eye(4)[:3]
    array = np.arange(4*5*6*3).reshape(4,5,6,3)
    expected_exception = ValueError
    match = "affine must be a square matrix."
    with pytest.raises(expected_exception, match=match):
        _multiply_coords_by_affine(affine, array)

    # Verify compatibility between affine and array.
    affine = np.eye(4)
    array = np.arange(4*5*6*2).reshape(4,5,6,2)
    expected_exception = ValueError
    match = "array is incompatible with affine. The length of the last dimension of array should be 1 less than the length of affine."
    with pytest.raises(expected_exception, match=match):
        _multiply_coords_by_affine(affine, array)

    # Verify warning on violation of homogenous coordinates in affine.
    
    affine = np.ones((4,4))
    array = np.arange(4*5*6*3).reshape(4,5,6,3)
    expected_warning = RuntimeWarning
    match = "affine is not in homogenous coordinates.\naffine\[-1] should be zeros with a 1 on the right."
    with pytest.warns(expected_warning, match=match):
        _multiply_coords_by_affine(affine, array)
    
    affine = np.zeros((4,4))
    array = np.arange(4*5*6*3).reshape(4,5,6,3)
    expected_warning = RuntimeWarning
    match = "affine is not in homogenous coordinates.\naffine\[-1] should be zeros with a 1 on the right."
    with pytest.warns(expected_warning, match=match):
        _multiply_coords_by_affine(affine, array)
 
"""
Test resample.
"""

def test_resample():

    # Test proper use.

    # Test upsample with origin='zero'.
    kwargs = dict(
        image=np.array([
            [0, 1, 2], 
            [3, 4, 5], 
        ]), 
        new_resolution=1/2, 
        old_resolution=1, 
        err_to_larger=True, 
        extrapolation_fill_value=None, 
        origin='zero', 
        method='linear', 
        anti_aliasing=False, 
    )
    correct_output = np.array([
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5], 
        [1.5, 2.0, 2.5, 3.0, 3.5, 4.0], 
        [3.0, 3.5, 4.0, 4.5, 5.0, 5.5], 
        [4.5, 5.0, 5.5, 6.0, 6.5, 7.0], 
    ])
    assert np.array_equal(resample(**kwargs), correct_output)

    # Test upsample with origin='center'.
    kwargs = dict(
        image=np.array([
            [0, 1, 2], 
            [3, 4, 5], 
        ]), 
        new_resolution=1/2, 
        old_resolution=1, 
        err_to_larger=True, 
        extrapolation_fill_value=None, 
        origin='center', 
        method='linear', 
        anti_aliasing=False, 
    )
    correct_output = np.array([
        [-1.0, -0.5,  0.0,  0.5,  1.0,  1.5], 
        [ 0.5,  1.0,  1.5,  2.0,  2.5,  3.0], 
        [ 2.0,  2.5,  3.0,  3.5,  4.0,  4.5], 
        [ 3.5,  4.0,  4.5,  5.0,  5.5,  6.0], 
    ])
    assert np.array_equal(resample(**kwargs), correct_output)

    # Test downsample with origin='zero' and anti_aliasing=False.
    kwargs = dict(
        image=np.array([
            [ 0,  1,  2,  3,  4], 
            [ 5,  6,  7,  8,  9], 
            [10, 11, 12, 13, 14], 
            [15, 16, 17, 18, 19], 
        ]), 
        new_resolution=2, 
        old_resolution=1, 
        err_to_larger=True, 
        extrapolation_fill_value=None, 
        origin='zero', 
        method='linear', 
        anti_aliasing=False, 
    )
    correct_output = np.array([
        [ 0,  2,  4], 
        [10, 12, 14], 
    ])
    assert np.array_equal(resample(**kwargs), correct_output)

    # Test downsample with origin='center' and anti_aliasing=False.
    kwargs = dict(
        image=np.array([
            [ 0,  1,  2,  3,  4], 
            [ 5,  6,  7,  8,  9], 
            [10, 11, 12, 13, 14], 
            [15, 16, 17, 18, 19], 
        ]), 
        new_resolution=2, 
        old_resolution=1, 
        err_to_larger=True, 
        extrapolation_fill_value=None, 
        origin='center', 
        method='linear', 
        anti_aliasing=False, 
    )
    correct_output = np.array([
        [ 2.5,  4.5,  6.5],
        [12.5, 14.5, 16.5], 
    ])
    assert np.array_equal(resample(**kwargs), correct_output)

    # Test joint upsample and downsample with origin='zero' and anti_aliasing=False.
    kwargs = dict(
        image=np.array([
            [0, 1, 2, 3, 4], 
            [5, 6, 7, 8, 9], 
        ]), 
        new_resolution=[1/2, 2], 
        old_resolution=1, 
        err_to_larger=True, 
        extrapolation_fill_value=None, 
        origin='zero', 
        method='linear', 
        anti_aliasing=False, 
    )
    correct_output = np.array([
        [   0,    2,    4], 
        [ 2.5,  4.5,  6.5], 
        [ 5.0,  7.0,  9.0], 
        [ 7.5,  9.5, 11.5], 
    ])
    assert np.array_equal(resample(**kwargs), correct_output)

    # Test joint upsample and downsample with origin='center' and anti_aliasing=False.
    kwargs = dict(
        image=np.array([
            [0, 1, 2, 3, 4], 
            [5, 6, 7, 8, 9], 
        ]), 
        new_resolution=[1/2, 2], 
        old_resolution=1, 
        err_to_larger=True, 
        extrapolation_fill_value=None, 
        origin='center', 
        method='linear', 
        anti_aliasing=False, 
    )
    correct_output = np.array([
        [-1.25,  0.75,  2.75], 
        [ 1.25,  3.25,  5.25], 
        [ 3.75,  5.75,  7.75], 
        [ 6.25,  8.25, 10.25], 
    ])
    assert np.array_equal(resample(**kwargs), correct_output)

    # Test joint upsample and downsample with origin='zero', anti_aliasing=False, and err_to_larger=False.
    kwargs = dict(
        image=np.array([
            [ 0,  1,  2,  3,  4], 
            [ 5,  6,  7,  8,  9], 
            [10, 11, 12, 13, 14], 
        ]), 
        new_resolution=[1/2, 2], 
        old_resolution=1, 
        err_to_larger=False, 
        extrapolation_fill_value=None, 
        origin='zero', 
        method='linear', 
        anti_aliasing=False, 
    )
    correct_output = np.array([
        [   0,    2], 
        [ 2.5,  4.5], 
        [ 5.0,  7.0], 
        [ 7.5,  9.5], 
        [10.0, 12.0], 
        [12.5, 14.5], 
    ])
    assert np.array_equal(resample(**kwargs), correct_output)

"""
Perform tests.
"""

if __name__ == "__main__":
    test__validate_scalar_to_multi()
    test__validate_ndarray()
    test__validate_resolution()
    test__compute_axes()
    test__compute_coords()
    test__multiply_coords_by_affine()
    test_resample()
