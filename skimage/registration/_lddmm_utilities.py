"""
Defines:
    Functions:
        _validate_scalar_to_multi(value, size=3, dtype=float)
        _validate_ndarray(array, minimum_ndim=0, required_ndim=None, dtype=None, 
            forbid_object_dtype=True, broadcast_to_shape=None)
        _validate_xyz_resolution(ndim, xyz_resolution)
        _compute_axes(shape, xyz_resolution=1, origin='center')
        _compute_coords(shape, xyz_resolution=1, origin='center')
        _multiply_by_affine(array, affine)
"""

import numpy as np

def _validate_scalar_to_multi(value, size=3, dtype=float):
    """
    If value's length is 1, upcast it to match size. 
    Otherwise, if it does not match size, raise error.

    Return a numpy array.
    """

    # Cast size to int.
    try:
        size = int(size)
    except (TypeError, ValueError):
        raise TypeError(f"size must be interpretable as an integer.\n"
            f"type(size): {type(size)}.")
    
    if size < 0:
        raise ValueError(f"size must be non-negative.\n"
            f"size: {size}.")
    
    # Cast value to np.ndarray.
    try:
        value = np.array(value, dtype)
    except ValueError:
        raise ValueError(f"value and dtype are incompatible with one another.")

    # Validate value's dimensionality and length.
    if value.ndim == 0:
        value = np.array([value])
    if value.ndim == 1:
        if len(value) == 1:
            # Upcast scalar to match size.
            value = np.full(size, value, dtype=dtype)
        elif len(value) != size:
            # value's length is incompatible with size.
            raise ValueError(f"The length of value must either be 1 or it must match size.\n"
                f"len(value): {len(value)}, size: {size}.")
    else:
        # value.ndim > 1.
        raise ValueError(f"value must not have more than 1 dimension.\n"
            f"value.ndim: {value.ndim}.")
    
    # TODO: verify that this is necessary and rewrite/remove accordingly.
    # Check for np.nan values.
    if np.any(np.isnan(value)):
        raise NotImplementedError("np.nan values encountered. What input led to this result?\n"
            "Write in an exception as appropriate.")
        raise ValueError(f"value contains inappropriate values for the chosen dtype "
            f"and thus contains np.nan values.")
            
    return value


def _validate_ndarray(array, minimum_ndim=0, required_ndim=None, dtype=None, 
forbid_object_dtype=True, broadcast_to_shape=None, reshape_to_shape=None):
    """Cast (a copy of) array to a np.ndarray if possible and return it 
    unless it is noncompliant with minimum_ndim, required_ndim, and dtype.
    
    Note:
    
    If required_ndim is None, _validate_ndarray will accept any object.
    If it is possible to cast to dtype, otherwise an exception is raised.

    If np.array(array).ndim == 0 and required_ndim == 1, array will be upcast to ndim 1.
    
    If forbid_object_dtype == True and the dtype is object, an exception is raised 
    unless object is the dtype.
    
    If a shape is provided to broadcast_to_shape, unless noncompliance is found with 
    required_ndim, array is broadcasted to that shape.
    
    if a shape is provided to reshape_to_shape, array is reshaped to that shape."""

    # Verify arguments.

    # Verify minimum_ndim.
    if not isinstance(minimum_ndim, int):
        raise TypeError(f"minimum_ndim must be of type int.\n"
            f"type(minimum_ndim): {type(minimum_ndim)}.")
    if minimum_ndim < 0:
        raise ValueError(f"minimum_ndim must be non-negative.\n"
            f"minimum_ndim: {minimum_ndim}.")

    # Verify required_ndim.
    if required_ndim is not None:
        if not isinstance(required_ndim, int):
            raise TypeError(f"required_ndim must be either None or of type int.\n"
                f"type(required_ndim): {type(required_ndim)}.")
        if required_ndim < 0:
            raise ValueError(f"required_ndim must be non-negative.\n"
                f"required_ndim: {required_ndim}.")

    # Verify dtype.
    if dtype is not None:
        if not isinstance(dtype, type):
            raise TypeError(f"dtype must be either None or a valid type.\n"
                f"type(dtype): {type(dtype)}.")

    # Validate array.

    # Cast array to np.ndarray.
    # Validate compliance with dtype.
    try:
        array = np.array(array, dtype) # Side effect: breaks alias.
    except TypeError:
        raise TypeError(f"array is of a type that is incompatible with dtype.\n"
            f"type(array): {type(array)}, dtype: {dtype}.")
    except ValueError:
        raise ValueError(f"array has a value that is incompatible with dtype.\n"
            f"array: {array}, \ntype(array): {type(array)}, dtype: {dtype}.")

    # Verify compliance with forbid_object_dtype.
    if forbid_object_dtype:
        if array.dtype == object and dtype != object:
            raise TypeError(f"Casting array to a np.ndarray produces an array of dtype object \n"
                f"while forbid_object_dtype == True and dtype != object.")

    # Validate compliance with required_ndim.
    if required_ndim is not None and array.ndim != required_ndim:
        # Upcast from ndim 0 to ndim 1 if appropriate.
        if array.ndim == 0 and required_ndim == 1:
            array = np.array([array])
        else:
            raise ValueError(f"If required_ndim is not None, array.ndim must equal it unless array.ndim == 0 and required_ndin == 1.\n"
                f"array.ndim: {array.ndim}, required_ndim: {required_ndim}.")

    # Verify compliance with minimum_ndim.
    if array.ndim < minimum_ndim:
        raise ValueError(f"array.ndim must be at least equal to minimum_ndim.\n"
            f"array.ndim: {array.ndim}, minimum_ndim: {minimum_ndim}.")
    
    # Broadcast array if appropriate.
    if broadcast_to_shape is not None:
        array = np.copy(np.broadcast_to(array=array, shape=broadcast_to_shape))

    # Reshape array if appropriate.
    if reshape_to_shape is not None:
        array = np.copy(array.reshape(reshape_to_shape))

    return array

# TODO: reverse order of arguments and propagate change throughout ardent.
def _validate_xyz_resolution(ndim, xyz_resolution):
    """Validate xyz_resolution to assure its length matches the dimensionality of image."""

    xyz_resolution = _validate_scalar_to_multi(xyz_resolution, size=ndim)

    if np.any(xyz_resolution <= 0):
        raise ValueError(f"All elements of xyz_resolution must be positive.\n"
            f"np.min(xyz_resolution): {np.min(xyz_resolution)}.")

    return xyz_resolution


def _compute_axes(shape, xyz_resolution=1, origin='center'):
    """Returns the real_axes defining an image with the given shape 
    at the given resolution as a list of numpy arrays.
    """

    # Validate shape.
    shape = _validate_ndarray(shape, dtype=int, required_ndim=1)

    # Validate xyz_resolution.
    xyz_resolution = _validate_xyz_resolution(len(shape), xyz_resolution)

    # Create axes.

    # axes is a list of arrays matching each shape element from shape, spaced by the corresponding xyz_resolution.
    axes = [np.arange(dim_size) * dim_res for dim_size, dim_res in zip(shape, xyz_resolution)]

    # List all presently recognized origin values.
    origins = ['center', 'zero']

    if origin == 'center':
        # Center each axes array to its mean.
        for xyz_index, axis in enumerate(axes):
            axes[xyz_index] -= np.mean(axis)
    elif origin == 'zero':
        # Allow each axis to increase from 0 along each dimension.
        pass
    else:
        raise NotImplementedError(f"origin must be one of these supported values: {origins}.\n"
            f"origin: {origin}.")
    
    return axes


def _compute_coords(shape, xyz_resolution=1, origin='center'):
    """Returns the real_coordinates of an image with the given shape 
    at the given resolution as a single numpy array of shape (*shape, len(shape))."""

    axes = _compute_axes(shape, xyz_resolution, origin)

    meshes = np.meshgrid(*axes, indexing='ij')

    return np.stack(meshes, axis=-1)


def _multiply_by_affine(array, affine, spatial_dimensions=3):

    arrays = []
    for dim in range(spatial_dimensions):
        arrays.append(np.sum(affine[dim, :-1] * array + affine[dim, -1], axis=-1))

    return np.stack(arrays=arrays, axis=-1)

    # Expanded for 3D:
    # return np.stack(
    #     arrays=[
    #         np.sum(affine[0, :3] * array + affine[0, -1], axis=-1), 
    #         np.sum(affine[1, :3] * array + affine[1, -1], axis=-1), 
    #         np.sum(affine[2, :3] * array + affine[2, -1], axis=-1), 
    #     ],
    #     axis=-1,
    # )
