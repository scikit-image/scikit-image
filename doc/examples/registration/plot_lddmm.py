import numpy as np

from matplotlib import pyplot as plt
from scipy.ndimage import map_coordinates
from skimage.registration import lddmm_register
from skimage.data import allen_mouse_brain_atlas, cleared_mouse_brain
from skimage.transform import resize, rescale


def scale_data(data, quantile_threshold=0.001):
    """
    Rescales data such that the central data points (ignoring the extrema) lie
    on the interval [0, 1].
    """

    data = np.copy(data)

    lower_limit = np.quantile(
        data, min(quantile_threshold, 1 - quantile_threshold)
    )
    upper_limit = np.quantile(
        data, max(quantile_threshold, 1 - quantile_threshold)
    )
    data_range = upper_limit - lower_limit

    data -= lower_limit
    data /= data_range

    return data


def imshow_on_ax(
    axes, dim, column, image, overlaid_image=None, quantile_threshold=0.001
):
    """
    Rescales image using scale_data and displays a central slice of it across
    the given dimension on the specified element of axes.

    If an overlaid_image is provided, it is likewise rescaled and an RGB
    display is produced using image as the Red and Blue channels and
    overlaid_image as the Green channel.
    """

    ax = axes[dim, column]
    ax.axis(False)

    scaled_image = scale_data(image, quantile_threshold)

    display_image = scaled_image

    if overlaid_image is not None:
        scaled_overlaid_image = scale_data(overlaid_image, quantile_threshold)
        display_image = np.stack(
            [scaled_image, scaled_overlaid_image, scaled_image], axis=-1
        )

    ax.imshow(
        display_image.take(display_image.shape[dim] // 2, axis=dim),
        cmap="gray",
        vmin=0,
        vmax=1,
    )


def generate_calibration_plots(
    affines,
    maximum_velocities,
    matching_energies,
    regularization_energies,
    total_energies,
):
    """
    Plot the energies, maximum velocities, translation components, and linear
    components as functions of the number of iterations.
    """

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    # Plot matching, regularization, and total energies.
    ax = axes[0, 0]
    ax.plot(
        list(zip(matching_energies, regularization_energies, total_energies))
    )
    ax.set_title("Energies")

    # Plot the maximum velocity.
    ax = axes[0, 1]
    ax.plot(maximum_velocities)
    ax.set_title("Maximum\nvelocity")

    # Plot affine[:, :-1], the translation components.
    translations = [affine[:-1, -1] for affine in affines]
    ax = axes[1, 0]
    ax.plot(translations)
    ax.set_title("Translation\ncomponents")

    # Plot self.affine[:-1, :-1], the linear transformation components.
    linear_components = [affine[:-1, :-1].ravel() for affine in affines]
    ax = axes[1, 1]
    ax.plot(linear_components)
    ax.set_title("Linear\ncomponents")


# Load images.
reference_image = allen_mouse_brain_atlas()
moving_image = cleared_mouse_brain()


# Specify spacings.
reference_image_spacing = np.array([100, 100, 100])
moving_image_spacing = np.array([100, 100, 100])


# Learn registration from reference_image to moving_image.
lddmm_output = lddmm_register(
    reference_image=reference_image,
    moving_image=moving_image,
    reference_image_spacing=reference_image_spacing,
    moving_image_spacing=moving_image_spacing,
    multiscales=[8, 4],
    affine_stepsize=0.3,
    deformative_stepsize=2e5,
    sigma_regularization=1e4,
    contrast_order=3,
    num_iterations=[50, 100],
    num_affine_only_iterations=[50, 0],
    num_rigid_affine_iterations=[25, 0],
)


# Visualize registration progression, useful for parameter tuning.
generate_calibration_plots(
    affines=lddmm_output.diagnostics.affines,
    maximum_velocities=lddmm_output.diagnostics.maximum_velocities,
    matching_energies=lddmm_output.diagnostics.matching_energies,
    regularization_energies=lddmm_output.diagnostics.regularization_energies,
    total_energies=lddmm_output.diagnostics.total_energies,
)

# Apply registration to reference_image and moving_image.
"""
The registration can be reasonably applied to any image in the reference_image
    or moving_image space by substituting it for the corresponding image
    (reference_image or moving_image) as the input argument of map_coordinates
    and multiplying the coordinates by
    different_image.shape / replaced_image.shape. 
    We demonstrate this by using rescaled versions of reference_image and
    moving_image to imitate different images in the same spaces.

The registration can be applied at arbitrary spacing by first multiplying
    the coordinates by the factor change in spacing and then resampling the
    coordinates to the desired spacing. 
    We demonstrate this by applying at the same spacing as our rescaled
    reference_image and moving_image to get deformed images of the same shape.
    
The position-fields (transforms) output by the registration have the same
    shape as (are at the spacing of) the reference_image and moving_image
    regardless of the scale the registration was computed at, see the
    multiscales parameter.
"""

apply = "at native spacing"
# apply = "to different images at native spacing"
# apply = "to different images at different spacing"


# Apply registration to reference_image and moving_image at native spacing.

if apply == "at native spacing":

    deformed_moving_image = map_coordinates(
        input=moving_image,
        coordinates=lddmm_output.moving_image_to_reference_image_transform,
    )

    deformed_reference_image = map_coordinates(
        input=reference_image,
        coordinates=lddmm_output.reference_image_to_moving_image_transform,
    )

    reference_image_vis = reference_image
    moving_image_vis = moving_image


# Apply to different images
# (mocked with rescaled versions of reference_image and moving_image)
# at native spacing.

elif apply == "to different images at native spacing":

    rescaled_reference_image = rescale(
        reference_image, np.pi
    )  # This could by any image in the reference_image space.
    rescaled_moving_image = rescale(
        moving_image, np.e
    )  # This could by any image in the moving_image space.

    reference_image_scale = np.divide(
        rescaled_reference_image.shape, reference_image.shape
    )
    moving_image_scale = np.divide(
        rescaled_moving_image.shape, moving_image.shape
    )

    deformed_moving_image = map_coordinates(
        input=rescaled_moving_image,
        coordinates=lddmm_output.moving_image_to_reference_image_transform
        * moving_image_scale[:, None, None, None],
    )

    deformed_reference_image = map_coordinates(
        input=rescaled_reference_image,
        coordinates=lddmm_output.reference_image_to_moving_image_transform
        * reference_image_scale[:, None, None, None],
    )

    reference_image_vis = reference_image
    moving_image_vis = moving_image


# Apply to different images
# (mocked with rescaled versions of reference_image and moving_image)
# at a different spacing
# (at the exact shapes of rescaled_reference_image and rescaled_moving_image).

# Note: The difference between this and the above example is that the
# transformation is applied at higher spacing therefore produces deformed
# images of higher spacing. In this case it is producing deformed images of
# the same shape as their undeformed counterparts, rescaled_reference_image
# and rescaled_moving_image.

elif apply == "to different images at different spacing":

    rescaled_reference_image = rescale(reference_image, np.pi)
    rescaled_moving_image = rescale(moving_image, np.e)

    reference_image_scale = np.divide(
        rescaled_reference_image.shape, reference_image.shape
    )
    moving_image_scale = np.divide(
        rescaled_moving_image.shape, moving_image.shape
    )

    rescaled_moving_image_to_reference_image_transform = resize(
        lddmm_output.moving_image_to_reference_image_transform
        * moving_image_scale[:, None, None, None],
        (3, *rescaled_reference_image.shape),
    )
    rescaled_reference_image_to_moving_image_transform = resize(
        lddmm_output.reference_image_to_moving_image_transform
        * reference_image_scale[:, None, None, None],
        (3, *rescaled_moving_image.shape),
    )

    deformed_moving_image = map_coordinates(
        input=rescaled_moving_image,
        coordinates=rescaled_moving_image_to_reference_image_transform,
    )

    deformed_reference_image = map_coordinates(
        input=rescaled_reference_image,
        coordinates=rescaled_reference_image_to_moving_image_transform,
    )

    reference_image_vis = rescaled_reference_image
    moving_image_vis = rescaled_moving_image


# Visualize results.

# Column 0: raw reference_image.
# Column 1: moving_image deformed to reference_image.
# Column 2: deformed_moving_image overlaid with reference_image.
# Column 3: deformed_reference_image overlaid with moving_image.
# Column 4: reference_image deformed to moving_image.
# Column 5: raw moving_image.

fig, axes = plt.subplots(3, 6, figsize=(16, 8))

fig.suptitle("Registration: Before & After")

# Call imshow for each subplot axes.
for dim in range(3):
    # vmin and vmax are set to saturate the top and bottom 0.1% extrema.

    # Column 0: raw reference_image.
    imshow_on_ax(axes=axes, dim=dim, column=0, image=reference_image_vis)

    # Column 1: deformed_moving_image.
    imshow_on_ax(axes=axes, dim=dim, column=1, image=deformed_moving_image)

    # Column 2: deformed_moving_image overlaid with reference_image.
    imshow_on_ax(
        axes=axes,
        dim=dim,
        column=2,
        image=deformed_moving_image,
        overlaid_image=reference_image_vis,
    )

    # Column 3: deformed_reference_image overlaid with moving_image.
    imshow_on_ax(
        axes=axes,
        dim=dim,
        column=3,
        image=deformed_reference_image,
        overlaid_image=moving_image_vis,
    )

    # Column 4: deformed_reference_image.
    imshow_on_ax(axes=axes, dim=dim, column=4, image=deformed_reference_image)

    # Column 5: raw moving_image.
    imshow_on_ax(axes=axes, dim=dim, column=5, image=moving_image_vis)


# Set column labels.
for ax, column_label in zip(
    axes[0],
    [
        "reference_image",
        "deformed_moving_image",
        "deformed_moving_image \n& reference_image overlay",
        "deformed_reference_image \n& moving_image overlay",
        "deformed_reference_image",
        "moving_image",
    ],
):
    ax.set_title(column_label)


# Set row labels.
for ax, row_index in zip(axes[:, 0], range(len(axes))):
    row_label = f"Dimension {row_index}"
    ax.set_ylabel(row_label, rotation="vertical")

