import numpy as np
import warnings

from scipy.linalg import inv
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter
from skimage._shared.fft import fftmodule


def _validate_scalar_to_multi(value, size=None, dtype=None, reject_nans=True):
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
                "size must be either None or interpretable as an integer.\n"
                f"type(size): {type(size)}."
            ) from exception

        if size < 0:
            raise ValueError(f"size must be non-negative.\n" f"size: {size}.")

    # Cast value to np.ndarray.
    try:
        value = np.array(value, dtype=dtype)
    except ValueError as exception:
        raise ValueError(
            "value and dtype are incompatible with one another.\n"
            f"type(value): {type(value)}, dtype: {dtype}."
        ) from exception

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
                "The length of value must either be 1 or it must match size "
                "if size is provided.\n"
                f"len(value): {len(value)}, size: {size}."
            )
    else:
        # value.ndim > 1.
        raise ValueError(
            "value must not have more than 1 dimension.\n"
            f"value.ndim: {value.ndim}."
        )

    # Check for np.nan values.
    if reject_nans and np.any(np.isnan(value)):
        raise ValueError("value contains np.nan elements.")

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

    If minimum_ndim is provided and array.ndim < minimum_ndim, array.shape is
    left-padded by 1's until minimum_ndim is satisfied.

    If required_ndim is provided and array.ndim != required_ndim, an exception
    is raised.

    If forbid_object_dtype == True and array.dtype == object, an exception is
    raised, unless dtype is provided as object.

    If a shape is provided to broadcast_to_shape, array is broadcasted to that
    shape.

    If a shape is provided to reshape_to_shape, array is reshaped to that
    shape.

    If a shape is provided to required_shape, if the shape does not match this
    shape then an exception is raised.
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
            "minimum_ndim must be non-negative.\n"
            f"minimum_ndim: {minimum_ndim}."
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
        array = np.array(array, dtype)  # Side effect: breaks alias.
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
                "Casting array to a np.ndarray produces an array of dtype "
                "object while forbid_object_dtype == True and dtype != "
                "object."
            )

    # Validate compliance with minimum_ndim by left-padding the shape with 1's
    # as necessary.
    if array.ndim < minimum_ndim:
        array = array.reshape(*[1] * (minimum_ndim - array.ndim), *array.shape)

    # Validate compliance with required_ndim.
    if required_ndim is not None and array.ndim != required_ndim:
        raise ValueError(
            "If required_ndim is not None, array.ndim must be made to equal "
            "it.\n"
            f"array.ndim: {array.ndim}, required_ndim: {required_ndim}."
        )

    # Broadcast array if appropriate.
    if broadcast_to_shape is not None:
        array = np.copy(
            np.broadcast_to(
                array=array,
                shape=broadcast_to_shape,
            )
        )

    # Reshape array if appropriate.
    if reshape_to_shape is not None:
        array = np.copy(array.reshape(reshape_to_shape))

    # Verify compliance with required_shape if appropriate.
    if required_shape is not None:
        try:
            required_shape_satisfied = np.array_equal(
                array.reshape(required_shape).shape,
                array.shape,
            )
        except ValueError as exception:
            raise ValueError(
                f"array is incompatible with required_shape.\n"
                f"array.shape: {array.shape}, "
                f"required_shape: {required_shape}."
            ) from exception
        if not required_shape_satisfied:
            raise ValueError(
                f"array is compatible with required_shape but "
                "does not match required_shape.\n"
                f"array.shape: {array.shape}, "
                f"required_shape: {required_shape}."
            )

    return array


def _validate_spacing(spacing, ndim, dtype=float):
    """
    Validate spacing to assure its length matches the dimensionality of
    image.
    """

    spacing = _validate_scalar_to_multi(spacing, size=ndim, dtype=dtype)

    if np.any(spacing <= 0):
        raise ValueError(
            f"All elements of spacing must be positive.\n"
            f"np.min(spacing): {np.min(spacing)}."
        )

    return spacing


def _compute_axes(shape, spacing=1, origin="center", dtype=float):
    """
    Returns the real_axes defining an image with the given shape
    at the given spacing as a list of numpy arrays.
    """

    # Validate shape.
    shape = _validate_ndarray(shape, dtype=int, required_ndim=1)

    # Validate spacing.
    spacing = _validate_spacing(spacing, len(shape))

    # Create axes.

    # axes is a list of arrays matching each shape element from shape, spaced
    # by the corresponding spacing.
    axes = [
        np.arange(dim_size, dtype=dtype) * dim_res
        for dim_size, dim_res in zip(shape, spacing)
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


def _compute_coords(shape, spacing=1, origin="center", dtype=float):
    """
    Returns the real_coordinates of an image with the given shape at the given
    spacing as a single numpy array of shape (*shape, len(shape)).
    """

    axes = _compute_axes(shape, spacing, origin, dtype)

    meshes = np.meshgrid(*axes, indexing="ij")

    return np.stack(meshes, axis=-1)


def _multiply_coords_by_affine(affine, array):
    """
    Applies affine to the elements of array at each spatial position and
    returns the result.
    """

    # Validate inputs.

    # Verify that affine is square.
    if affine.ndim != 2:
        raise ValueError(
            f"affine must be a 2-dimensional matrix.\n"
            f"affine.ndim: {affine.ndim}."
        )
    # affine is 2-dimensional.
    if affine.shape[0] != affine.shape[1]:
        raise ValueError(
            f"affine must be a square matrix.\n"
            f"affine.shape: {affine.shape}."
        )
    # affine is square.

    # Verify compatibility between affine and array.
    if array.shape[-1] != len(affine) - 1:
        raise ValueError(
            "array is incompatible with affine. The length of the last "
            "dimension of array should be 1 less than the length of affine.\n"
            f"array.shape: {array.shape}, affine.shape: {affine.shape}."
        )

    # Raise warning if affine is not in homogenous coordinates.
    if not np.array_equal(
        affine[-1],
        np.array([0] * (len(affine) - 1) + [1]),
    ):
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
    """
    Computes and returns the determinant of array along its last 2 dimensions.
    """

    # Validate that array is square on its last 2 dimensions.
    if array.shape[-1] != array.shape[-2]:
        raise ValueError(
            f"array must be square on its last 2 dimensions.\n"
            f"array.shape[-2:]: {array.shape[-2:]}."
        )

    # Compute the determinant recursively.

    if array.shape[-1] == 2:
        # Handle 2-dimensional base case.
        determinant = (
            array[..., 0, 0] * array[..., 1, 1]
            - array[..., 0, 1] * array[..., 1, 0]
        )
    else:
        # Handle more than 2-dimensional recursive case.
        determinant = 0
        for dim in range(array.shape[-1]):
            recursive_indices = list(range(array.shape[-1]))
            recursive_indices.remove(dim)
            determinant += (
                (-1) ** dim
                * array[..., 0, dim]
                * _compute_tail_determinant(array[..., 1:, recursive_indices])
            )

    return determinant


def resample(
    image,
    new_spacing,
    old_spacing=1,
    err_to_larger=True,
    extrapolation_fill_value=None,
    origin="center",
    method="linear",
    image_is_coords=False,
    anti_aliasing=True,
):
    """
    Resamples image from an old spacing to a new spacing.

    Parameters
    ----------
    image: np.ndarray
        The image to be resampled
    new_spacing: float, seq
        The spacing of the resampled image.
    old_spacing: float, seq, optional
        The spacing of the input image. By default 1.
    err_to_larger: bool, optional
        Determines whether to round the new shape up or down.
        By default True.
    extrapolation_fill_value: float, optional
        The fill_value kwarg passed to interpn. By default None.
    origin: str, optional
        The origin to use for the image axes and coordinates used
        internally. By default 'center'.
    method: str, optional
        The method of interpolation, passed as the method kwarg in
        interpn. By default 'linear'.
    image_is_coords: bool, optional
        If True, this implies that the last dimension of image is not a
        spatial dimension and not subject to interpolation.
        By default False.
    anti_aliasing: bool, optional
        If True, applies a gaussian filter across dimensions to be
        downsampled before interpolating. By default True.

    Returns
    -------
    np.ndarray
        The result of resampling image at new_spacing.

    """

    # Validate inputs and define ndim & old_shape based on image_is_coords.
    image = _validate_ndarray(image)  # Breaks alias.
    if image_is_coords:
        ndim = image.ndim - 1
        old_shape = image.shape[:-1]
    else:
        ndim = image.ndim
        old_shape = image.shape
    new_spacing = _validate_spacing(new_spacing, ndim)
    old_spacing = _validate_spacing(old_spacing, ndim)

    # Handle trivial case.
    if np.array_equal(new_spacing, old_spacing):
        # Note: this is a copy of the input image and is not the same object.
        return image

    # Compute new_coords and old_axes.
    if err_to_larger:
        new_shape = np.ceil(old_shape * old_spacing / new_spacing)
    else:
        new_shape = np.floor(old_shape * old_spacing / new_spacing)
    new_coords = _compute_coords(new_shape, new_spacing, origin)
    old_axes = _compute_axes(old_shape, old_spacing, origin)

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
    Resample array to new_shape by padding and truncating at high frequencies
    in the fourier domain.

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
    new_shape = _validate_ndarray(
        new_shape, dtype=int, required_ndim=1, required_shape=array.ndim
    )

    resampled_array = np.copy(array)

    for dim in range(array.ndim):
        fourier_transformed_array = fftmodule.rfft(resampled_array, axis=dim)
        resampled_array = fftmodule.irfft(
            fourier_transformed_array, axis=dim, n=new_shape[dim]
        )
    resampled_array *= resampled_array.size / array.size

    return resampled_array


def generate_position_field(
    affine,
    velocity_fields,
    velocity_field_spacing,
    reference_image_shape,
    reference_image_spacing,
    moving_image_shape,
    moving_image_spacing,
    deform_to="reference_image",
):
    """
    Integrate velocity_fields and apply affine to produce a position field.

    Parameters
    ----------
    affine : np.ndarray
        The affine array to be incorporated into the returned position field.
    velocity_fields : np.ndarray
        The velocity_fields defining the diffeomorphic flow. The leading
        dimensions are spatial, and the last two dimensions are the number of
        time steps and the coordinates.
    velocity_field_spacing : float, seq
        The spacing of velocity_fields, with multiple values given to
        specify anisotropy.
    reference_image_shape : seq
        The shape of the reference_image.
    reference_image_spacing : float, seq
        The spacing of the reference_image, with multiple values given to
        specify anisotropy.
    moving_image_shape : seq
        The shape of the moving_image.
    moving_image_spacing : float, seq
        The spacing of the moving_image, with multiple values given to
        specify anisotropy.
    deform_to : str, optional
        The direction of the deformation. By default "reference_image".

    Returns
    -------
    np.ndarray
        The position field for the registration in the space of the image
        specified by deform_to.

    Raises
    ------
    ValueError
        Raised if the leading dimensions of velocity_fields fail to match
        reference_image_shape.
    TypeError
        Raised if deform_to is not of type str.
    ValueError
        Raised if deform_to is neither 'reference_image' nor 'moving_image'.
    """

    # Validate inputs.
    # Validate reference_image_shape. Not rigorous.
    reference_image_shape = _validate_ndarray(reference_image_shape)
    # Validate moving_image_shape. Not rigorous.
    moving_image_shape = _validate_ndarray(moving_image_shape)
    # Validate velocity_fields.
    velocity_fields = _validate_ndarray(
        velocity_fields, required_ndim=len(reference_image_shape) + 2
    )
    if not np.all(velocity_fields.shape[:-2] == reference_image_shape):
        raise ValueError(
            "velocity_fields' initial dimensions must equal "
            "reference_image_shape.\n"
            f"velocity_fields.shape: {velocity_fields.shape}, "
            f"reference_image_shape: {reference_image_shape}."
        )
    # Validate velocity_field_spacing.
    velocity_field_spacing = _validate_spacing(
        velocity_field_spacing, velocity_fields.ndim - 2
    )
    # Validate affine.
    affine = _validate_ndarray(
        affine,
        required_ndim=2,
        reshape_to_shape=(
            len(reference_image_shape) + 1,
            len(reference_image_shape) + 1,
        ),
    )
    # Verify deform_to.
    if not isinstance(deform_to, str):
        raise TypeError(
            f"deform_to must be of type str.\n"
            f"type(deform_to): {type(deform_to)}."
        )
    elif deform_to not in ["reference_image", "moving_image"]:
        raise ValueError(
            "deform_to must be either 'reference_image'" "or 'moving_image'."
        )

    # Compute intermediates.
    num_timesteps = velocity_fields.shape[-2]
    delta_t = 1 / num_timesteps
    reference_image_axes = _compute_axes(
        reference_image_shape,
        reference_image_spacing,
    )
    reference_image_coords = _compute_coords(
        reference_image_shape,
        reference_image_spacing,
    )
    moving_image_axes = _compute_axes(
        moving_image_shape,
        moving_image_spacing,
    )
    moving_image_coords = _compute_coords(
        moving_image_shape,
        moving_image_spacing,
    )

    # Create position field.
    if deform_to == "reference_image":
        phi = np.copy(reference_image_coords)
    elif deform_to == "moving_image":
        phi_inv = np.copy(reference_image_coords)

    # Integrate velocity field.
    for timestep in (
        reversed(range(num_timesteps))
        if deform_to == "reference_image"
        else range(num_timesteps)
    ):
        if deform_to == "reference_image":
            sample_coords = (
                reference_image_coords
                + velocity_fields[..., timestep, :] * delta_t
            )
            phi = (
                interpn(
                    points=reference_image_axes,
                    values=phi - reference_image_coords,
                    xi=sample_coords,
                    bounds_error=False,
                    fill_value=None,
                )
                + sample_coords
            )
        elif deform_to == "moving_image":
            sample_coords = (
                reference_image_coords
                - velocity_fields[..., timestep, :] * delta_t
            )
            phi_inv = (
                interpn(
                    points=reference_image_axes,
                    values=phi_inv - reference_image_coords,
                    xi=sample_coords,
                    bounds_error=False,
                    fill_value=None,
                )
                + sample_coords
            )

    # Apply the affine transform to the position field.
    if deform_to == "reference_image":
        # Apply the affine by multiplication.
        affine_phi = _multiply_coords_by_affine(affine, phi)
        # affine_phi has the spacing of the reference_image.
    elif deform_to == "moving_image":
        # Apply the affine by interpolation.
        sample_coords = _multiply_coords_by_affine(
            inv(affine),
            moving_image_coords,
        )
        phi_inv_affine_inv = (
            interpn(
                points=reference_image_axes,
                values=phi_inv - reference_image_coords,
                xi=sample_coords,
                bounds_error=False,
                fill_value=None,
            )
            + sample_coords
        )
        # phi_inv_affine_inv has the spacing of the moving_image.

    # return appropriate position field.
    if deform_to == "reference_image":
        return affine_phi
    elif deform_to == "moving_image":
        return phi_inv_affine_inv
