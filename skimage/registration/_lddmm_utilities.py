"""
Defines:
    Internal functions:
        _validate_scalar_to_multi(value, size=None, dtype=None)
        _validate_ndarray(array, dtype=None, minimum_ndim=0, required_ndim=None, 
            forbid_object_dtype=True, broadcast_to_shape=None, reshape_to_shape=None, required_shape=None)
        _validate_resolution(resolution, ndim)
        _compute_axes(shape, resolution=1, origin='center')
        _compute_coords(shape, resolution=1, origin='center')
        _multiply_coords_by_affine(array, affine)
        _compute_tail_determinant(array)
    User functions:
        resample(image, new_resolution, old_resolution=1, 
            err_to_larger=True, extrapolation_fill_value=None, 
            origin='center', method='linear', image_is_coords=False)
        sinc_resample(array, new_shape)
"""

import numpy as np
import warnings

from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter


def _validate_scalar_to_multi(value, size=None, dtype=None):
    """
    If value's length is 1, upcast it to match size. 
    Otherwise, if it does not match size, raise error.

    If size is not provided, cast to a 1-dimensional np.ndarray.

    Return a numpy array.
    """

    # Cast size to int if appropriate.
    if size is not None:
        try:
            size = int(size)
        except (TypeError, ValueError) as exception:
            raise TypeError(
                f"size must be either None or interpretable as an integer.\n" f"type(size): {type(size)}."
            ) from exception

        if size < 0:
            raise ValueError(f"size must be non-negative.\n" f"size: {size}.")

    # Cast value to np.ndarray.
    try:
        value = np.array(value, dtype=dtype)
    except ValueError as exception:
        raise ValueError(f"value and dtype are incompatible with one another.") from exception

    # Validate value's dimensionality and length.
    if value.ndim == 0:
        value = np.array([value.item()])
    if value.ndim == 1:
        if size is not None and len(value) == 1:
            # Upcast scalar to match size.
            value = np.full(size, value, dtype=dtype)
        elif size is not None and len(value) != size:
            # value's length is incompatible with size.
            raise ValueError(
                f"The length of value must either be 1 or it must match size if size is provided.\n"
                f"len(value): {len(value)}, size: {size}."
            )
    else:
        # value.ndim > 1.
        raise ValueError(
            f"value must not have more than 1 dimension.\n" f"value.ndim: {value.ndim}."
        )

    # Check for np.nan values if the dtype is not object.
    if value.dtype != 'object' and np.any(np.isnan(value)):
        raise NotImplementedError(
            "np.nan values encountered for a value not cast to dtype object. What input led to this result?\n"
            "Write in an exception as appropriate."
        )
        raise ValueError(
            f"value contains inappropriate values for the chosen dtype "
            f"and thus contains np.nan values."
        )

    return value


def _validate_ndarray(
    array,
    dtype=None,
    forbid_object_dtype=True,
    minimum_ndim=1,
    required_ndim=None,
    broadcast_to_shape=None,
    reshape_to_shape=None,
    required_shape=None,
):
    """
    Cast (a copy of) array to a np.ndarray if possible and return it 
    unless it is noncompliant with minimum_ndim, required_ndim, and dtype.
    
    Note: the following checks and validations are performed in order.
    
    If required_ndim is None or 0, _validate_ndarray will accept any object.
    
    array is cast to an np.ndarray of type dtype.

    If minimum_ndim is provided and array.ndim < minimum_ndim, array.shape is left-padded by 1's until minimum_ndim is satisfied.

    If required_ndim is provided and array.ndim != required_ndim, an exception is raised.

    If forbid_object_dtype == True and array.dtype == object, an exception is raised, unless dtype is provided as object.
    
    If a shape is provided to broadcast_to_shape, array is broadcasted to that shape.
    
    If a shape is provided to reshape_to_shape, array is reshaped to that shape.

    If a shape is provided to required_shape, if the shape does not match this shape then an exception is raised.
    """

    # Verify arguments.

    # Verify minimum_ndim.
    if not isinstance(minimum_ndim, int):
        raise TypeError(
            f"minimum_ndim must be of type int.\n"
            f"type(minimum_ndim): {type(minimum_ndim)}."
        )
    if minimum_ndim < 0:
        raise ValueError(
            f"minimum_ndim must be non-negative.\n" f"minimum_ndim: {minimum_ndim}."
        )

    # Verify required_ndim.
    if required_ndim is not None:
        if not isinstance(required_ndim, int):
            raise TypeError(
                f"required_ndim must be either None or of type int.\n"
                f"type(required_ndim): {type(required_ndim)}."
            )
        if required_ndim < 0:
            raise ValueError(
                f"required_ndim must be non-negative.\n"
                f"required_ndim: {required_ndim}."
            )

    # Verify dtype.
    if dtype is not None:
        if not isinstance(dtype, type):
            raise TypeError(
                f"dtype must be either None or a valid type.\n"
                f"type(dtype): {type(dtype)}."
            )

    # Validate array.

    # Cast array to np.ndarray.
    # Validate compliance with dtype.
    try:
        array = np.array(array, dtype) # Side effect: breaks alias.
    except TypeError as exception:
        raise TypeError(
            f"array is of a type that is incompatible with dtype.\n"
            f"type(array): {type(array)}, dtype: {dtype}."
        ) from exception
    except ValueError as exception:
        raise ValueError(
            f"array has a value that is incompatible with dtype.\n"
            f"array: {array}, \ntype(array): {type(array)}, dtype: {dtype}."
        ) from exception

    # Verify compliance with forbid_object_dtype.
    if forbid_object_dtype:
        if array.dtype == object and dtype != object:
            raise TypeError(
                f"Casting array to a np.ndarray produces an array of dtype object \n"
                f"while forbid_object_dtype == True and dtype != object."
            )

    # Validate compliance with minimum_ndim by left-padding the shape with 1's as necessary.
    if array.ndim < minimum_ndim:
        array = array.reshape(*[1] * (minimum_ndim - array.ndim), *array.shape)

    # Validate compliance with required_ndim.
    if required_ndim is not None and array.ndim != required_ndim:
        raise ValueError(
            f"If required_ndim is not None, array.ndim must be made to equal it.\n"
            f"array.ndim: {array.ndim}, required_ndim: {required_ndim}."
        )

    # Broadcast array if appropriate.
    if broadcast_to_shape is not None:
        array = np.copy(np.broadcast_to(array=array, shape=broadcast_to_shape))

    # Reshape array if appropriate.
    if reshape_to_shape is not None:
        array = np.copy(array.reshape(reshape_to_shape))

    # Verify compliance with required_shape if appropriate.
    if required_shape is not None:
        try:
            required_shape_satisfied = np.array_equal(array.reshape(required_shape).shape, array.shape)
        except ValueError as exception:
            raise ValueError(
                f"array is incompatible with required_shape.\n"
                f"array.shape: {array.shape}, required_shape: {required_shape}."
            ) from exception
        if not required_shape_satisfied:
            raise ValueError(f"array is compatible with required_shape but does not match required_shape.\n"
                             f"array.shape: {array.shape}, required_shape: {required_shape}.")

    return array


def _validate_resolution(resolution, ndim, dtype=float):
    """Validate resolution to assure its length matches the dimensionality of image."""

    resolution = _validate_scalar_to_multi(resolution, size=ndim, dtype=dtype)

    if np.any(resolution <= 0):
        raise ValueError(
            f"All elements of resolution must be positive.\n"
            f"np.min(resolution): {np.min(resolution)}."
        )

    return resolution


def _compute_axes(shape, resolution=1, origin="center"):
    """
    Returns the real_axes defining an image with the given shape 
    at the given resolution as a list of numpy arrays.
    """

    # Validate shape.
    shape = _validate_ndarray(shape, dtype=int, required_ndim=1)

    # Validate resolution.
    resolution = _validate_resolution(resolution, len(shape))

    # Create axes.

    # axes is a list of arrays matching each shape element from shape, spaced by the corresponding resolution.
    axes = [
        np.arange(dim_size) * dim_res for dim_size, dim_res in zip(shape, resolution)
    ]

    # List all presently recognized origin values.
    origins = ["center", "zero"]

    if origin == "center":
        # Center each axes array to its mean.
        for axis_index, axis in enumerate(axes):
            axes[axis_index] -= np.mean(axis)
    elif origin == "zero":
        # Allow each axis to increase from 0 along each dimension.
        pass
    else:
        raise NotImplementedError(
            f"origin must be one of these supported values: {origins}.\n"
            f"origin: {origin}."
        )

    return axes


def _compute_coords(shape, resolution=1, origin="center"):
    """
    Returns the real_coordinates of an image with the given shape 
    at the given resolution as a single numpy array of shape (*shape, len(shape)).
    """

    axes = _compute_axes(shape, resolution, origin)

    meshes = np.meshgrid(*axes, indexing="ij")

    return np.stack(meshes, axis=-1)


def _multiply_coords_by_affine(affine, array):
    """Applies affine to the elements of array at each spatial position and returns the result."""

    # Validate inputs.

    # Verify that affine is square.
    if affine.ndim != 2:
        raise ValueError(
            f"affine must be a 2-dimensional matrix.\n" f"affine.ndim: {affine.ndim}."
        )
    # affine is 2-dimensional.
    if affine.shape[0] != affine.shape[1]:
        raise ValueError(
            f"affine must be a square matrix.\n" f"affine.shape: {affine.shape}."
        )
    # affine is square.

    # Verify compatibility between affine and array.
    if array.shape[-1] != len(affine) - 1:
        raise ValueError(
            f"array is incompatible with affine. The length of the last dimension of array should be 1 less than the length of affine.\n"
            f"array.shape: {array.shape}, affine.shape: {affine.shape}."
        )

    # Raise warning if affine is not in homogenous coordinates.
    if not np.array_equal(affine[-1], np.array([0] * (len(affine) - 1) + [1])):
        warnings.warn(
            message=f"affine is not in homogenous coordinates.\n"
                    f"affine[-1] should be zeros with a 1 on the right.\n"
                    f"affine[-1]: {affine[-1]}.",
            category=RuntimeWarning,
        )

    # Perform affine matrix multiplication.

    ndims = len(affine) - 1
    return (
        np.squeeze(np.matmul(affine[:ndims, :ndims], array[..., None]), -1)
        + affine[:ndims, ndims]
    )


def _compute_tail_determinant(array):
    """Computes and returns the determinant of array along its last 2 dimensions."""

    # Validate that array is square on its last 2 dimensions.
    if array.shape[-1] != array.shape[-2]:
        raise ValueError(f"array must be square on its last 2 dimensions.\n"
                         f"array.shape[-2:]: {array.shape[-2:]}.")

    # Compute the determinant recursively.

    if array.shape[-1] == 2:
        # Handle 2-dimensional base case.
        determinant = array[...,0,0] * array[...,1,1] - array[...,0,1] * array[...,1,0]
    else:
        # Handle more than 2-dimensional recursive case.
        determinant = 0
        for dim in range(array.shape[-1]):
            recursive_indices = list(range(array.shape[-1]))
            recursive_indices.remove(dim)
            determinant += (-1)**dim * array[...,0,dim] * _compute_tail_determinant(array[...,1:,recursive_indices])
    
    return determinant


def resample(
    image,
    new_resolution,
    old_resolution=1,
    err_to_larger=True,
    extrapolation_fill_value=None,
    origin="center",
    method="linear",
    image_is_coords=False,
    anti_aliasing=True,
):
    """
    Resamples image from an old resolution to a new resolution.
    
    Parameters
    ----------
        image: np.ndarray
            The image to be resampled
        new_resolution: float, seq
            The resolution of the resampled image.
        old_resolution: float, seq, optional
            The resolution of the input image. By default 1.
        err_to_larger: bool, optional
            Determines whether to round the new shape up or down. By default True.
        extrapolation_fill_value: float, optional
            The fill_value kwarg passed to interpn. By default None.
        origin: str, optional
            The origin to use for the image axes and coordinates used internally. By default 'center'.
        method: str, optional
            The method of interpolation, passed as the method kwarg in interpn. By default 'linear'.
        image_is_coords: bool, optional
            If True, this implies that the last dimension of image is not a spatial dimension and not subject to interpolation. By default False.
        anti_aliasing: bool, optional
            If True, applies a gaussian filter across dimensions to be downsampled before interpolating. By default True.
    
    Returns
    -------
    np.ndarray
        The result of resampling image at new_resolution.
    """

    # Validate inputs and define ndim & old_shape based on image_is_coords.
    image = _validate_ndarray(image) # Breaks alias.
    if image_is_coords:
        ndim = image.ndim - 1
        old_shape = image.shape[:-1]
    else:
        ndim = image.ndim
        old_shape = image.shape
    new_resolution = _validate_resolution(new_resolution, ndim)
    old_resolution = _validate_resolution(old_resolution, ndim)

    # Handle trivial case.
    if np.array_equal(new_resolution, old_resolution):
        return (
            image # Note: this is a copy of the input image and is not the same object.
        )

    # Compute new_coords and old_axes.
    if err_to_larger:
        new_shape = np.ceil(old_shape * old_resolution / new_resolution)
    else:
        new_shape = np.floor(old_shape * old_resolution / new_resolution)
    new_coords = _compute_coords(new_shape, new_resolution, origin)
    old_axes = _compute_axes(old_shape, old_resolution, origin)

    # Apply anti-aliasing gaussian filter if downsampling.
    if anti_aliasing:
        if image_is_coords:
            downsample_factors = np.insert(
                old_shape / new_shape, image.ndim - 1, values=0, axis=0
            )
        else:
            downsample_factors = old_shape / new_shape
        anti_aliasing_sigma = np.maximum(0, (downsample_factors - 1) / 2)
        gaussian_filter(
            image, anti_aliasing_sigma, output=image, mode="nearest"
        )  # Mutates image.

    # Interpolate image.
    new_image = interpn(
        points=old_axes,
        values=image,
        xi=new_coords,
        bounds_error=False,
        fill_value=extrapolation_fill_value,
    )

    return new_image


def sinc_resample(array, new_shape):
    """
    Resample array to new_shape by padding and truncating at high frequencies in the fourier domain.

    Parameters
    ----------
    array : np.ndarray
        The array to be resampled.
    new_shape : seq
        The shape to resample array to.

    Returns
    -------
    np.ndarray
        A copy of array after resampling.
    """

    # Validate inputs.
    array = _validate_ndarray(array)
    new_shape = _validate_ndarray(new_shape, dtype=int, required_ndim=1, required_shape=array.ndim)

    resampled_array = np.copy(array)

    for dim in range(array.ndim):
        fourier_transformed_array = np.fft.rfft(resampled_array, axis=dim)
        resampled_array = np.fft.irfft(fourier_transformed_array, axis=dim, n=new_shape[dim])
    resampled_array *= resampled_array.size / array.size

    return resampled_array
