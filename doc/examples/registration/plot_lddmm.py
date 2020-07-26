import numpy as np
from matplotlib import pyplot as plt

from skimage.registration import lddmm_register
from skimage.data import allen_mouse_brain_atlas, cleared_mouse_brain
from skimage.transform import resize, rescale


def scale_data(data, quantile_threshold=0.001):
    """Rescales data such that the central data points (ignoring the extrema) lie on the interval [0, 1]."""
    
    data = np.copy(data)
    
    lower_limit = np.quantile(data, min(quantile_threshold, 1 - quantile_threshold))
    upper_limit = np.quantile(data, max(quantile_threshold, 1 - quantile_threshold))
    data_range = upper_limit - lower_limit
    
    data -= lower_limit
    data /= data_range
    
    return data


def imshow_on_ax(axes, dim, column, image, overlaid_image=None, quantile_threshold=0.001):
    """
    Rescales image using scale_data and displays a central slice of it across the given dimension 
    on the specified element of <axes>.

    If an overlaid_image is provided, it is likewise rescaled and an RGB display is produced 
    using image as the Red and Blue channels and overlaid_image as the Green channel.
    """
    
    ax = axes[dim, column]
    ax.axis(False)
    
    scaled_image = scale_data(image, quantile_threshold)
    
    display_image = scaled_image
    
    if overlaid_image is not None:
        scaled_overlaid_image = scale_data(overlaid_image, quantile_threshold)
        display_image = np.stack([scaled_image, scaled_overlaid_image, scaled_image], axis=-1)
        
    ax.imshow(
        display_image.take(display_image.shape[dim] // 2, axis=dim), 
        cmap='gray', 
        vmin=0, 
        vmax=1, 
    )
    

def generate_calibration_plots(affines, maximum_velocities, matching_energies, regularization_energies, total_energies):
    """Plot the energies, maximum velocities, translation components, and linear components as functions of the number of iterations."""
    
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    # Plot matching, regularization, and total energies.
    ax = axes[0, 0]
    ax.plot(list(zip(matching_energies, regularization_energies, total_energies)))
    ax.set_title('Energies')

    # Plot the maximum velocity.
    ax = axes[0, 1]
    ax.plot(maximum_velocities)
    ax.set_title('Maximum\nvelocity')

    # Plot affine[:, :-1], the translation components.
    translations = [affine[:-1, -1] for affine in affines]
    ax = axes[1, 0]
    ax.plot(translations)
    ax.set_title('Translation\ncomponents')

    # Plot self.affine[:-1, :-1], the linear transformation components.
    linear_components = [affine[:-1, :-1].ravel() for affine in affines]
    ax = axes[1, 1]
    ax.plot(linear_components)
    ax.set_title('Linear\ncomponents')


# Load images.
template = allen_mouse_brain_atlas()
target = cleared_mouse_brain()


# Specify resolutions.
template_resolution = np.array([100, 100, 100])
target_resolution = np.array([100, 100, 100])


# Learn registration from template to target.
lddmm_output = lddmm_register(
    template                    = template,
    target                      = target,
    template_resolution         = template_resolution,
    target_resolution           = target_resolution,
    multiscales                 = [8, 4],
    affine_stepsize             = 0.3,
    deformative_stepsize        = 2e5,
    sigma_regularization        = 1e4,
    contrast_order              = 3,
    num_iterations              = [50, 100],
    num_affine_only_iterations  = [50, 0],
    num_rigid_affine_iterations = [25, 0],
    # track_progress_every_n=10 prints a progress update every 10 iterations of registration. Pending removal.
    track_progress_every_n      = 10,
)


# Visualize registration progression, useful for parameter tuning.
generate_calibration_plots(
    affines = lddmm_output.diagnostics.affines, 
    maximum_velocities = lddmm_output.diagnostics.maximum_velocities, 
    matching_energies = lddmm_output.diagnostics.matching_energies, 
    regularization_energies = lddmm_output.diagnostics.regularization_energies, 
    total_energies = lddmm_output.diagnostics.total_energies,
)

# Apply registration to template and target.
"""
The registration can be reasonably applied to any image in the template or target space
    by substituting it for the corresponding image (template or target) as the input argument of map_coordinates 
    and multiplying the coordinates by different_image.shape / replaced_image.shape.

The registration can be applied at arbitrary resolution by first multiplying the coordinates by the factor change in resolution 
    and then resampling the coordinates to the desired resolution.
    
The position-fields (transforms) output by the registration have the same shape as (are at the resolution of) the template and target 
    regardless of the scale the registration was computed at, see the multiscales parameter.
"""

apply = "at native resolution"
# apply = "to different images at native resolution"
# apply = "to different images at different resolution"


# Apply registration to template and target at native resolution.

if apply == "at native resolution":

    deformed_target = map_coordinates(
        input=target,
        coordinates=lddmm_output.target_to_template_transform,
    )

    deformed_template = map_coordinates(
        input=template,
        coordinates=lddmm_output.template_to_target_transform,
    )

    template_vis = template
    target_vis = target

    
# Apply to different images (mocked with rescaled versions of template and target) at native resolution.

if apply == "to different images at native resolution":

    rescaled_template = rescale(template, np.pi) # This could by any image in the template space.
    rescaled_target = rescale(target, np.e) # This could by any image in the target space.

    template_scale = np.divide(rescaled_template.shape, template.shape)
    target_scale = np.divide(rescaled_target.shape, target.shape)

    deformed_target = map_coordinates(
        input=rescaled_target,
        coordinates=lddmm_output.target_to_template_transform * target_scale[:, None, None, None],
    )

    deformed_template = map_coordinates(
        input=rescaled_template,
        coordinates=lddmm_output.template_to_target_transform * template_scale[:, None, None, None],
    )

    template_vis = template
    target_vis = target

    
# Apply to different images (mocked with rescaled versions of template and target) 
# at a different resolution (at the exact shapes of rescaled_template and rescaled_target). 

# Note: The difference between this and the above example is that the transformation is applied at higher resolution 
      # therefore produces deformed images of higher resolution. 
      # In this case it is producing deformed images of the same shape as their undeformed counterparts, rescaled_template and rescaled_target.

if apply == "to different images at different resolution":
    
    rescaled_template = rescale(template, np.pi)
    rescaled_target = rescale(target, np.e)

    template_scale = np.divide(rescaled_template.shape, template.shape)
    target_scale = np.divide(rescaled_target.shape, target.shape)

    rescaled_target_to_template_transform = resize(lddmm_output.target_to_template_transform * target_scale[:, None, None, None], (3, *rescaled_template.shape))
    rescaled_template_to_target_transform = resize(lddmm_output.template_to_target_transform * template_scale[:, None, None, None], (3, *rescaled_target.shape))

    deformed_target = map_coordinates(
        input=rescaled_target,
        coordinates=rescaled_target_to_template_transform,
    )

    deformed_template = map_coordinates(
        input=rescaled_template,
        coordinates=rescaled_template_to_target_transform,
    )

    template_vis = rescaled_template
    target_vis = rescaled_target


# Visualize results.

# Column 0: raw template.
# Column 1: target deformed to template.
# Column 2: deformed_target overlaid with template.
# Column 3: deformed_template overlaid with target.
# Column 4: template deformed to target.
# Column 5: raw target.

fig, axes = plt.subplots(3, 6, figsize=(16,8))#, sharex=True, sharey=True)

fig.suptitle('Registration: Before & After')

# Call imshow for each subplot axes.
for dim in range(3):
    # vmin and vmax are set to saturate the top and bottom 0.1% extrema.
    
    # Column 0: raw template.
    imshow_on_ax(axes=axes, dim=dim, column=0, image=template_vis)
    
    # Column 1: deformed_target.
    imshow_on_ax(axes=axes, dim=dim, column=1, image=deformed_target)
    
    # Column 2: deformed_target overlaid with template.
    imshow_on_ax(axes=axes, dim=dim, column=2, image=deformed_target, overlaid_image=template_vis)

    # Column 3: deformed_template overlaid with target.
    imshow_on_ax(axes=axes, dim=dim, column=3, image=deformed_template, overlaid_image=target_vis)

    # Column 4: deformed_template.
    imshow_on_ax(axes=axes, dim=dim, column=4, image=deformed_template)
    
    # Column 5: raw target.
    imshow_on_ax(axes=axes, dim=dim, column=5, image=target_vis)
    

# Set column labels.
for ax, column_label in zip(axes[0], [
        'template', 
        'deformed_target', 
        'deformed_target \n& template overlay', 
        'deformed_template \n& target overlay', 
        'deformed_template', 
        'target', 
    ]):
    ax.set_title(column_label)
    

# Set row labels.
for ax, row_index in zip(axes[:, 0], range(len(axes))):
    row_label = f'Dimension {row_index}'
    ax.set_ylabel(row_label, rotation='vertical')
