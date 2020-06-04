import pytest

import numpy as np

from skimage.registration._lddmm_utilities import _validate_scalar_to_multi
from skimage.registration._lddmm_utilities import _validate_ndarray
from skimage.registration._lddmm_utilities import _validate_resolution
from skimage.registration._lddmm_utilities import _compute_axes
from skimage.registration._lddmm_utilities import _compute_coords
from skimage.registration._lddmm_utilities import _multiply_coords_by_affine
from skimage.registration._lddmm_utilities import resample
from skimage.registration._lddmm_utilities import sinc_resample

"""
Test _validate_scalar_to_multi.
"""

class Test__validate_scalar_to_multi:

    # Test proper use.

    def test_singular(self):
        kwargs = dict(value=1, size=1, dtype=float)
        correct_output = np.array([1], float)
        assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    def test_empty(self):
        kwargs = dict(value=1, size=0, dtype=int)
        correct_output = np.array([], int)
        assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    def test_scalar_dtype_casting(self):
        kwargs = dict(value=9.5, size=4, dtype=int)
        correct_output = np.full(4, 9, int)
        assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    def test_list_value_dtype_casting_to_float(self):
        kwargs = dict(value=[1, 2, 3.5], size=3, dtype=float)
        correct_output = np.array([1, 2, 3.5], float)
        assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    def test_list_value_dtype_casting_to_int(self):
        kwargs = dict(value=[1, 2, 3.5], size=3, dtype=int)
        correct_output = np.array([1, 2, 3], int)
        assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    def test_tuple_value(self):
        kwargs = dict(value=(1, 2, 3), size=3, dtype=int)
        correct_output = np.array([1, 2, 3], int)
        assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    def test_array_value(self):
        kwargs = dict(value=np.array([1, 2, 3], float), size=3, dtype=int)
        correct_output = np.array([1, 2, 3], int)
        assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    def test_multi_value_with_size_None(self):
        kwargs = dict(value=np.array([1, 2, 3], float), size=None, dtype=float)
        correct_output = np.array([1, 2, 3], float)
        assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)


    def test_scalar_value_with_size_None(self):
        kwargs = dict(value=1, size=None, dtype=float)
        correct_output = np.array([1], float)
        assert np.array_equal(_validate_scalar_to_multi(**kwargs), correct_output)

    # Test improper use.

    def test_non_int_size(self):
        kwargs = dict(value=[1, 2, 3, 4], size='size: not an int', dtype=float)
        expected_exception = TypeError
        match = "size must be either None or interpretable as an integer."
        with pytest.raises(expected_exception, match=match):
            _validate_scalar_to_multi(**kwargs)

    def test_negative_size(self):
        kwargs = dict(value=[], size=-1, dtype=float)
        expected_exception = ValueError
        match = "size must be non-negative."
        with pytest.raises(expected_exception, match=match):
            _validate_scalar_to_multi(**kwargs)

    def test_value_incompatible_with_size(self):
        kwargs = dict(value=[1, 2, 3, 4], size=3, dtype=int)
        expected_exception = ValueError
        match = "The length of value must either be 1 or it must match size if size is provided."
        with pytest.raises(expected_exception, match=match):
            _validate_scalar_to_multi(**kwargs)

    def test_multi_dimensional_value(self):
        kwargs = dict(value=np.arange(3*4, dtype=int).reshape(3,4), size=3, dtype=float)
        expected_exception = ValueError
        match = "value must not have more than 1 dimension."
        with pytest.raises(expected_exception, match=match):
            _validate_scalar_to_multi(**kwargs)

    def test_list_value_uncastable_to_dtype(self):
        kwargs = dict(value=[1, 2, 'c'], size=3, dtype=int)
        expected_exception = ValueError
        match = "value and dtype are incompatible with one another."
        with pytest.raises(expected_exception, match=match):
            _validate_scalar_to_multi(**kwargs)

    def test_scalar_value_uncastable_to_dtype(self):
        kwargs = dict(value='c', size=3, dtype=int)
        expected_exception = ValueError
        match = "value and dtype are incompatible with one another."
        with pytest.raises(expected_exception, match=match):
            _validate_scalar_to_multi(**kwargs)

"""
Test _validate_ndarray.
"""

class Test__validate_ndarray:

    # Test proper use.

    def test_cast_array_to_dtype(self):
        kwargs = dict(array=np.arange(3, dtype=int), dtype=float)
        correct_output = np.arange(3, dtype=float)
        assert np.array_equal(_validate_ndarray(**kwargs), correct_output)

    def test_array_as_list(self):
        kwargs = dict(array=[[0,1,2], [3,4,5]], dtype=float)
        correct_output = np.arange(2*3, dtype=float).reshape(2,3)
        assert np.array_equal(_validate_ndarray(**kwargs), correct_output)

    def test_scalar_array_upcast_by_minimum_ndim(self):
        kwargs = dict(array=9, minimum_ndim=2)
        correct_output = np.array(9).reshape(1, 1)
        assert np.array_equal(_validate_ndarray(**kwargs), correct_output)

    def test_broadcast_to_shape(self):
        kwargs = dict(array=np.array([0,1,2]), broadcast_to_shape=(2,3))
        correct_output = np.array([[0,1,2], [0,1,2]])
        assert np.array_equal(_validate_ndarray(**kwargs), correct_output)

    def test_reshape_to_shape(self):
        kwargs = dict(array=np.arange(3 * 4).reshape(3, 4), reshape_to_shape=(4, 3))
        correct_output = np.arange(3 * 4).reshape(4, 3)
        assert np.array_equal(_validate_ndarray(**kwargs), correct_output)
    
    def test_wildcard_reshape_to_shape(self):
        kwargs = dict(array=np.arange(3 * 4).reshape(3, 4), reshape_to_shape=(4, -1))
        correct_output = np.arange(3 * 4).reshape(4, 3)
        assert np.array_equal(_validate_ndarray(**kwargs), correct_output)

    def test_wildcard_required_shape(self):
        kwargs = dict(array=np.arange(3 * 4).reshape(3, 4), required_shape=(3, -1))
        correct_output = np.arange(3 * 4).reshape(3, 4)
        assert np.array_equal(_validate_ndarray(**kwargs), correct_output)

    # Test improper use.

    # Validate arguments.

    def test_non_int_minimum_ndim(self):
        kwargs = dict(array=np.arange(3), minimum_ndim=1.5)
        expected_exception = TypeError
        match = "minimum_ndim must be of type int."
        with pytest.raises(expected_exception, match=match):
            _validate_ndarray(**kwargs)

    def test_negative_minimum_ndim(self):
        kwargs = dict(array=np.arange(3), minimum_ndim=-1)
        expected_exception = ValueError
        match = "minimum_ndim must be non-negative."
        with pytest.raises(expected_exception, match=match):
            _validate_ndarray(**kwargs)

    def test_non_int_required_ndim(self):
        kwargs = dict(array=np.arange(3), required_ndim=1.5)
        expected_exception = TypeError
        match = "required_ndim must be either None or of type int."
        with pytest.raises(expected_exception, match=match):
            _validate_ndarray(**kwargs)

    def test_negative_required_ndim(self):
        kwargs = dict(array=np.arange(3), required_ndim=-1)
        expected_exception = ValueError
        match = "required_ndim must be non-negative."
        with pytest.raises(expected_exception, match=match):
            _validate_ndarray(**kwargs)

    def test_non_type_dtype(self):
        kwargs = dict(array=np.arange(3), dtype="not of type type")
        expected_exception = TypeError
        match = "dtype must be either None or a valid type."
        with pytest.raises(expected_exception, match=match):
            _validate_ndarray(**kwargs)
        
    # Validate array.

    def test_array_type_incompataible_with_dtype(self):
        kwargs = dict(array=np.array(print), dtype=int)
        expected_exception = TypeError
        match = "array is of a type that is incompatible with dtype."
        with pytest.raises(expected_exception, match=match):
            _validate_ndarray(**kwargs)

    def test_array_value_incompatible_with_dtype(self):
        kwargs = dict(array=np.array('string that is not an int'), dtype=int)
        expected_exception = ValueError
        match = "array has a value that is incompatible with dtype."
        with pytest.raises(expected_exception, match=match):
            _validate_ndarray(**kwargs)
        
    def test_forbid_object_dtype(self):
        kwargs = dict(array=np.array([[], 1]), dtype=None, forbid_object_dtype=True)
        expected_exception = TypeError
        match = "Casting array to a np.ndarray produces an array of dtype object \nwhile forbid_object_dtype == True and dtype != object."
        with pytest.raises(expected_exception, match=match):
            _validate_ndarray(**kwargs)

    def test_required_ndim(self):
        kwargs = dict(array=np.arange(3), required_ndim=2)
        expected_exception = ValueError
        match = "If required_ndim is not None, array.ndim must be made to equal it."
        with pytest.raises(expected_exception, match=match):
            _validate_ndarray(**kwargs)

    def test_incompaatible_required_shape(self):
        kwargs = dict(array=np.arange(3 * 4).reshape(3, 4), required_shape=(3, 5))
        expected_exception = ValueError
        match = "array is incompatible with required_shape."
        with pytest.raises(expected_exception, match=match):
            _validate_ndarray(**kwargs)

    def test_compatible_but_not_matched_required_shape(self):
        kwargs = dict(array=np.arange(3 * 4).reshape(3, 4), required_shape=(4, -1))
        expected_exception = ValueError
        match = "array is compatible with required_shape but does not match required_shape."
        with pytest.raises(expected_exception, match=match):
            _validate_ndarray(**kwargs)

"""
Test _validate_resolution.
"""

class Test__validate_resolution:

    # Test proper use.
    
    def test_scalar_resolution_1D_ndim(self):
        kwargs = dict(resolution=2, ndim=1, dtype=float)
        correct_output = np.full(1, 2, float)
        assert np.array_equal(_validate_resolution(**kwargs), correct_output)

    def test_scalar_resolution_4D_ndim(self):
        kwargs = dict(resolution=1.5, ndim=4, dtype=float)
        correct_output = np.full(4, 1.5, float)
        assert np.array_equal(_validate_resolution(**kwargs), correct_output)

    def test_array_resolution_3D_ndim(self):
        kwargs = dict(resolution=np.ones(3, int), ndim=3, dtype=float)
        correct_output = np.ones(3, float)
        assert np.array_equal(_validate_resolution(**kwargs), correct_output)

    def test_list_resolution_2D_ndim(self):
        kwargs = dict(resolution=[3, 4], ndim=2, dtype=float)
        correct_output = np.array([3, 4], float)
        assert np.array_equal(_validate_resolution(**kwargs), correct_output)

        # Test improper use.

    def test_negative_resolution(self):
        kwargs = dict(resolution=[3, -4], ndim=2, dtype=float)
        expected_exception = ValueError
        match = "All elements of resolution must be positive."
        with pytest.raises(expected_exception, match=match):
            _validate_resolution(**kwargs)

    def test_zero_resolution(self):
        kwargs = dict(resolution=[3, 0], ndim=2, dtype=float)
        expected_exception = ValueError
        match = "All elements of resolution must be positive."
        with pytest.raises(expected_exception, match=match):
            _validate_resolution(**kwargs)

"""
Test _compute_axes.
"""

class Test__compute_axes:

    # Test proper use.

    # _compute_axes produces a list with a np.ndarray for each element in shape.

    def test_zero_and_one_in_shape(self):
        kwargs = dict(shape=(0, 1, 2), resolution=1, origin='center')
        correct_output = [np.arange(dim_size) * dim_res - np.mean(np.arange(dim_size) * dim_res) 
            for dim_size, dim_res in zip((0, 1, 2), (1, 1, 1))]
        for dim, coord in enumerate(_compute_axes(**kwargs)):
            assert np.array_equal(coord, correct_output[dim])

    def test_decimal_resolution(self):
        kwargs = dict(shape=(1, 2, 3, 4), resolution=1.5, origin='center')
        correct_output = [np.arange(dim_size) * dim_res - np.mean(np.arange(dim_size) * dim_res) 
            for dim_size, dim_res in zip((1, 2, 3, 4), (1.5, 1.5, 1.5, 1.5))]
        for dim, coord in enumerate(_compute_axes(**kwargs)):
            assert np.array_equal(coord, correct_output[dim])

    def test_anisotropic_resolution(self):
        kwargs = dict(shape=(2, 3, 4), resolution=[1, 1.5, 2], origin='center')
        correct_output = [np.arange(dim_size) * dim_res - np.mean(np.arange(dim_size) * dim_res) 
            for dim_size, dim_res in zip((2, 3, 4), (1, 1.5, 2))]
        for dim, coord in enumerate(_compute_axes(**kwargs)):
            assert np.array_equal(coord, correct_output[dim])

    def test_1D_shape(self):
        kwargs = dict(shape=5, resolution=1, origin='center')
        correct_output = [np.arange(dim_size) * dim_res - np.mean(np.arange(dim_size) * dim_res) 
            for dim_size, dim_res in zip((5,), (1,))]
        for dim, coord in enumerate(_compute_axes(**kwargs)):
            assert np.array_equal(coord, correct_output[dim])

    def test_zero_origin(self):
        kwargs = dict(shape=5, resolution=1, origin='zero')
        correct_output = [np.arange(dim_size) * dim_res
            for dim_size, dim_res in zip((5,), (1,))]
        for dim, coord in enumerate(_compute_axes(**kwargs)):
            assert np.array_equal(coord, correct_output[dim])

"""
Test _compute_coords.
"""

class Test__compute_coords:

    # Test proper use.

    def test_1D_shape_center_origin(self):
        kwargs = dict(shape=5, resolution=1, origin='center')
        correct_output = np.array([[-2], [-1], [0], [1], [2]])
        assert np.array_equal(_compute_coords(**kwargs), correct_output)

    def test_2D_shape_zero_origin(self):
        kwargs = dict(shape=(3,4), resolution=1, origin='zero')
        correct_output = np.array([[[0,0], [0,1], [0,2], [0,3]], [[1,0], [1,1], [1,2], [1,3]], [[2,0], [2,1], [2,2], [2,3]]])
        assert np.array_equal(_compute_coords(**kwargs), correct_output)

"""
Test _multiply_coords_by_affine.
"""

class Test__multiply_coords_by_affine:

    # Test proper use.

    def test_identity_affine_3D(self):
        affine = np.eye(4) + np.append(np.arange(3*4).reshape(3,4), np.zeros((1,4)), 0)**2
        array = _compute_coords((3,4,5), 1)
        result = _multiply_coords_by_affine(affine, array)
        arrays = []
        for dim in range(3):
            arrays.append(np.sum(affine[dim, :-1] * array, axis=-1) + affine[dim, -1])
        expected = np.stack(arrays=arrays, axis=-1)
        assert np.array_equal(result, expected)

    # Test improper use.

    def test_3D_affine_array(self):
        affine = np.eye(4)[None]
        array = np.arange(4*5*6*3).reshape(4,5,6,3)
        expected_exception = ValueError
        match = "affine must be a 2-dimensional matrix."
        with pytest.raises(expected_exception, match=match):
            _multiply_coords_by_affine(affine, array)

    def test_non_square_affine(self):
        affine = np.eye(4)[:3]
        array = np.arange(4*5*6*3).reshape(4,5,6,3)
        expected_exception = ValueError
        match = "affine must be a square matrix."
        with pytest.raises(expected_exception, match=match):
            _multiply_coords_by_affine(affine, array)

    def test_incompatible_affine_and_array(self):
        affine = np.eye(4)
        array = np.arange(4*5*6*2).reshape(4,5,6,2)
        expected_exception = ValueError
        match = "array is incompatible with affine. The length of the last dimension of array should be 1 less than the length of affine."
        with pytest.raises(expected_exception, match=match):
            _multiply_coords_by_affine(affine, array)

    # Verify warning on violation of homogenous coordinates in affine.
    
    def test_homogenous_coordinates_violation_warning(self):
        affine = np.ones((4,4))
        array = np.arange(4*5*6*3).reshape(4,5,6,3)
        expected_warning = RuntimeWarning
        match = "affine is not in homogenous coordinates.\naffine\[-1] should be zeros with a 1 on the right."
        with pytest.warns(expected_warning, match=match):
            _multiply_coords_by_affine(affine, array)
 
"""
Test resample.
"""

class Test_resample:

    # Test proper use.

    def test_zero_origin_upsample(self):
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

    def test_center_origin_upsample(self):
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

    def test_zero_origin_downsample(self):
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

    def test_center_origin_downsample(self):
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

    def test_zero_origin_joint_upsample_and_downsample(self):
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

    def test_center_origin_joint_upsample_and_downsample(self):
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

    def test_center_origin_joint_upsample_and_downsample_with_err_to_larger_False(self):
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
Test sinc_resample.
"""
# @pytest.mark.skip
class Test_sinc_resample:

    # Test shape of an upsample.
    def test_upsample_shape(self):
        kwargs = dict(
            array=np.arange(10*20*30).reshape(10,20,30),
            new_shape=(12, 45, 30),
        )
        output = sinc_resample(**kwargs)
        assert np.array_equal(output.shape, kwargs['new_shape'])

    # Test shape of a downsample.
    def test_downsample_shape(self):
        kwargs = dict(
            array=np.arange(10*20*30).reshape(10,20,30),
            new_shape=(8, 20, 14),
        )
        output = sinc_resample(**kwargs)
        assert np.array_equal(output.shape, kwargs['new_shape'])

    # Test shape of a general resample.
    def test_joint_upsample_and_downsample_shape(self):
        kwargs = dict(
            array=np.arange(10*20*30).reshape(10,20,30),
            new_shape=(10, 23, 27),
        )
        output = sinc_resample(**kwargs)
        assert np.array_equal(output.shape, kwargs['new_shape'])

    # Test preservation of the trends in a general resample.
    def test_preserves_trends(self):
        kwargs = dict(
            array=np.array([
                [0, 1, 2, 3, 4, 3, 2, 1, 0],
                [2, 3, 4, 5, 6, 5, 4, 3, 2],
                [0, 1, 2, 3, 4, 3, 2, 1, 0],
            ]), # Shape: (3, 9).
            new_shape=(5, 7),
        )
        output = sinc_resample(**kwargs)
        # Check that the first half of rows are ascending.
        for row in range((output.shape[0] - 1) // 2):
            assert np.all(output[row, :] <= output[row + 1, :])
        # Check that the second half of rows are descending.
        for row in range(output.shape[0] // 2, output.shape[0] - 1):
            assert np.all(output[row, :] >= output[row + 1, :])
        # Check that the first half of columns are ascending.
        for col in range((output.shape[1] - 1) // 2):
            assert np.all(output[:, col] <= output[:, col + 1])
        # Check that the second half of columns are descending.
        for col in range(output.shape[1] // 2, output.shape[1] - 1):
            assert np.all(output[:, col] >= output[:, col + 1])
        

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
    test_sinc_resample()
