import numpy as np
from matplotlib import pyplot as plt

from skimage.registration import lddmm_register, apply_lddmm, resample
from skimage.data import allen_mouse_brain_atlas, cleared_mouse_brain


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


# Load images.
template = allen_mouse_brain_atlas()
target = cleared_mouse_brain()


# Downsample images if desired.
# Note: with downsample_factor = 1, lddmm_register runs in approximately 40-50 minutes.
# This runtime declines roughly as the cube of downsample_factor.
# Optimization parameters may require adjustment for sufficiently large downsample_factor values.
downsample_factor = 2
template = resample(template, downsample_factor)
target   = resample(  target, downsample_factor)


# Learn registration from template to target.
# The stepsizes for lddmm_register are calibrated for this data without any downsampling (downsample_factor = 1 above), hence the correction.
print(
    "Please allow approximately 3-4 minutes for lddmm_register to finish with these inputs,\n"
    "or ~15-20 minutes if running with a single CPU."
)
lddmm_dict = lddmm_register(
    template                   = template, 
    target                     = target, 
    translational_stepsize     = 1e-5 * downsample_factor, 
    linear_stepsize            = 1e-8 * downsample_factor,
    deformative_stepsize       = 5e-1 * downsample_factor,
    sigmaR                     = 2e1,
    contrast_order             = 3, 
    num_iterations             = 100, 
    num_affine_only_iterations = 50, 
    # calibrate=True outputs a diagnostic plot at the end of lddmm_register that is useful for determining appropriate stepsizes.
    calibrate                  = False, 
)


# Apply registration to template and target.
# Note: the registration can be reasonably applied to any image in the template or target space
# by substituting it for the corresponding image (template or target) as the subject kwarg in apply_lddmm.

target_extrapolation_fill_value = np.quantile(target, 0.001)
template_extrapolation_fill_value = np.quantile(template, 0.001)

deformed_target = apply_lddmm(subject=target, deform_to='template', extrapolation_fill_value=target_extrapolation_fill_value, **lddmm_dict)

deformed_template = apply_lddmm(subject=template, deform_to='target', extrapolation_fill_value=template_extrapolation_fill_value, **lddmm_dict)


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
    imshow_on_ax(axes=axes, dim=dim, column=0, image=template)
    
    # Column 1: deformed_target.
    imshow_on_ax(axes=axes, dim=dim, column=1, image=deformed_target)
    
    # Column 2: deformed_target overlaid with template.
    imshow_on_ax(axes=axes, dim=dim, column=2, image=deformed_target, overlaid_image=template)

    # Column 3: deformed_template overlaid with target.
    imshow_on_ax(axes=axes, dim=dim, column=3, image=deformed_template, overlaid_image=target)

    # Column 4: deformed_template.
    imshow_on_ax(axes=axes, dim=dim, column=4, image=deformed_template)
    
    # Column 5: raw target.
    imshow_on_ax(axes=axes, dim=dim, column=5, image=target)
    

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
