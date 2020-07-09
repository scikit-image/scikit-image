"""
This is an implementation of the LDDMM algorithm with modifications, based on 
"Diffeomorphic registration with intensity transformation and missing data: Application to 3D digital pathology of Alzheimer's disease." 
This paper extends on an older LDDMM paper, 
"Computing large deformation metric mappings via geodesic flows of diffeomorphisms."

This is the more recent paper:
Tward, Daniel, et al. "Diffeomorphic registration with intensity transformation and missing data: Application to 3D digital pathology of Alzheimer's disease." Frontiers in neuroscience 14 (2020).
https://doi.org/10.3389/fnins.2020.00052

This is the original LDDMM paper:
Beg, M. Faisal, et al. "Computing large deformation metric mappings via geodesic flows of diffeomorphisms." International journal of computer vision 61.2 (2005): 139-157.
https://doi.org/10.1023/B:VISI.0000043755.93987.aa
"""

import warnings
import numpy as np
from scipy.interpolate import interpn
from scipy.linalg import inv, solve, det, svd
from scipy.sparse.linalg import cg, LinearOperator
from skimage.transform import resize, rescale
from matplotlib import pyplot as plt

from ._lddmm_utilities import _validate_ndarray
from ._lddmm_utilities import _validate_scalar_to_multi
from ._lddmm_utilities import _validate_resolution
from ._lddmm_utilities import _compute_axes
from ._lddmm_utilities import _compute_coords
from ._lddmm_utilities import _multiply_coords_by_affine
from ._lddmm_utilities import _compute_tail_determinant
from ._lddmm_utilities import resample
from ._lddmm_utilities import sinc_resample

r'''
  _            _       _                         
 | |          | |     | |                        
 | |        __| |   __| |  _ __ ___    _ __ ___  
 | |       / _` |  / _` | | '_ ` _ \  | '_ ` _ \ 
 | |____  | (_| | | (_| | | | | | | | | | | | | |
 |______|  \__,_|  \__,_| |_| |_| |_| |_| |_| |_|
                                                 
'''

#TODO: resolution --> spacing, template_resolution --> template_spacing.
#TODO: template --> reference_image, target --> moving_image.
#TODO: create issue that moving_image rudely implies one of two equal uses of a registration.
#TODO: explore replacing my lddmm_transform_[image/points] with scipy.ndimage.map_coordinates, allowable by converting position_fields to relative vector fields and putting the coordinates at the front of the shape.
#TODO: add attributes used to docstrings, check for natural groupings.
class _Lddmm:
    """
    Class for storing shared values and objects used in registration and performing the registration via methods.
    Accessed in a functional manner via the lddmm_register function; it instantiates an _Lddmm object and calls its register method.
    """

    def __init__(
        self,
        # Images.
        template,
        target,
        # Image resolutions.
        template_resolution=None,
        target_resolution=None,
        # Iterations.
        num_iterations=None,
        num_affine_only_iterations=None,
        num_rigid_affine_iterations=None,
        # Stepsizes.
        affine_stepsize=None,
        deformative_stepsize=None,
        # Affine specifiers.
        fixed_affine_scale=None,
        # Velocity field specifiers.
        sigma_regularization=None,
        velocity_smooth_length=None,
        preconditioner_velocity_smooth_length=None,
        maximum_velocity_fields_update=None,
        num_timesteps=None,
        # Contrast map specifiers.
        contrast_order=None,
        spatially_varying_contrast_map=None,
        contrast_maxiter=None,
        contrast_tolerance=None,
        sigma_contrast=None,
        contrast_smooth_length=None,
        # Smoothness vs. accuracy tradeoff.
        sigma_matching=None,
        # Classification specifiers.
        classify_and_weight_voxels=None,
        sigma_artifact=None,
        sigma_background=None,
        artifact_prior=None,
        background_prior=None,
        # Initial values.
        initial_affine=None,
        initial_contrast_coefficients=None,
        initial_velocity_fields=None,
        # Diagnostic outputs.
        calibrate=None,
        track_progress_every_n=None,
    ):

        # Constant inputs.

        # Images.
        self.template = _validate_ndarray(template, dtype=float)
        self.target = _validate_ndarray(target, dtype=float, required_ndim=self.template.ndim)

        # Resolution.
        self.template_resolution = _validate_scalar_to_multi(template_resolution if template_resolution is not None else 1, self.template.ndim, float)
        self.target_resolution = _validate_scalar_to_multi(target_resolution if target_resolution is not None else 1, self.target.ndim, float)

        # Iterations.
        self.num_iterations = int(num_iterations) if num_iterations is not None else 300
        self.num_affine_only_iterations = int(num_affine_only_iterations) if num_affine_only_iterations is not None else 100
        self.num_rigid_affine_iterations = int(num_rigid_affine_iterations) if num_rigid_affine_iterations is not None else 50

        # Stepsizes.
        self.affine_stepsize = float(affine_stepsize) if affine_stepsize is not None else 0.3
        self.deformative_stepsize = float(deformative_stepsize) if deformative_stepsize is not None else 0

        # Affine specifiers.
        self.fixed_affine_scale = float(fixed_affine_scale) if fixed_affine_scale is not None else None

        # Velocity field specifiers.
        self.sigma_regularization = float(sigma_regularization) if sigma_regularization is not None else 10 * np.max(self.template_resolution)
        self.velocity_smooth_length = float(velocity_smooth_length) if velocity_smooth_length is not None else 2 * np.max(self.template_resolution)
        self.preconditioner_velocity_smooth_length = float(preconditioner_velocity_smooth_length) if preconditioner_velocity_smooth_length is not None else 5 * np.max(self.template_resolution)
        self.maximum_velocity_fields_update = float(maximum_velocity_fields_update) if maximum_velocity_fields_update is not None else 1
        self.num_timesteps = int(num_timesteps) if num_timesteps is not None else 5

        # Contrast map specifiers.
        self.contrast_order = int(contrast_order) if contrast_order else 1
        if self.contrast_order < 1: raise ValueError(f"contrast_order must be at least 1.\ncontrast_order: {self.contrast_order}")
        self.spatially_varying_contrast_map = bool(spatially_varying_contrast_map) if spatially_varying_contrast_map is not None else False
        self.contrast_maxiter = int(contrast_maxiter) if contrast_maxiter else 5
        self.contrast_tolerance = float(contrast_tolerance) if contrast_tolerance else 1e-5
        self.sigma_contrast = float(sigma_contrast) if sigma_contrast else 1
        self.contrast_smooth_length = float(contrast_smooth_length) if contrast_smooth_length else 10 * np.max(self.target_resolution)

        # Smoothness vs. accuracy tradeoff.
        self.sigma_matching = float(sigma_matching) if sigma_matching else np.std(self.target)

        # Classification specifiers.
        self.classify_and_weight_voxels = bool(classify_and_weight_voxels) if classify_and_weight_voxels is not None else False
        self.sigma_artifact = float(sigma_artifact) if sigma_artifact else 5 * self.sigma_matching
        self.sigma_background = float(sigma_background) if sigma_background else 2 * self.sigma_matching
        self.artifact_prior = float(artifact_prior) if artifact_prior is not None else 1/3
        self.background_prior = float(background_prior) if background_prior is not None else 1/3
        if self.artifact_prior + self.background_prior >= 1:
            raise ValueError(f"artifact_prior and background_prior must sum to less than 1.")

        # Diagnostic outputs.
        self.calibrate = bool(calibrate) if calibrate is not None else False
        self.track_progress_every_n = int(track_progress_every_n) if track_progress_every_n is not None else 0

        # Constructions.

        # Constants.
        self.template_axes = _compute_axes(self.template.shape, self.template_resolution)
        self.template_coords = _compute_coords(self.template.shape, self.template_resolution)
        self.target_axes = _compute_axes(self.target.shape, self.target_resolution)
        self.target_coords = _compute_coords(self.target.shape, self.target_resolution)
        self.artifact_mean_value = np.max(self.target)
        self.background_mean_value = np.min(self.target)
        self.delta_t = 1 / self.num_timesteps
        self.fourier_filter_power = 2
        fourier_velocity_fields_coords = _compute_coords(self.template.shape, 1 / (self.template_resolution * self.template.shape), origin='zero')
        self.fourier_high_pass_filter = (
            1 - self.velocity_smooth_length**2 
            * np.sum((-2 + 2 * np.cos(2 * np.pi * fourier_velocity_fields_coords * self.template_resolution)) / self.template_resolution**2, axis=-1)
        )**self.fourier_filter_power
        fourier_template_coords = _compute_coords(self.template.shape, 1 / (self.template_resolution * self.template.shape), origin='zero')
        self.low_pass_filter = 1 / (
            (1 - self.velocity_smooth_length**2 * (
                np.sum((-2 + 2 * np.cos(2 * np.pi * self.template_resolution * fourier_template_coords)) / self.template_resolution**2, -1)
                )
            )**(2 * self.fourier_filter_power)
        )
        # This filter affects the optimization but not the optimum.
        self.preconditioner_low_pass_filter = 1 / (
            (1 - self.preconditioner_velocity_smooth_length**2 * (
                np.sum((-2 + 2 * np.cos(2 * np.pi * self.template_resolution * fourier_template_coords)) / self.template_resolution**2, -1)
                )
            )**(2 * self.fourier_filter_power)
        )
        fourier_target_coords = _compute_coords(self.target.shape, 1 / (self.target_resolution * self.target.shape), origin='zero')
        self.contrast_high_pass_filter = (
            1 - self.contrast_smooth_length**2 * (
                np.sum((-2 + 2 * np.cos(2 * np.pi * self.target_resolution * fourier_target_coords)) / self.target_resolution**2, -1)
            )
        )**self.fourier_filter_power / self.sigma_contrast

        # Dynamics.
        if initial_affine is None:
            initial_affine = np.eye(template.ndim + 1)
        self.affine = _validate_ndarray(initial_affine, required_shape=(self.template.ndim + 1, self.template.ndim + 1))
        if initial_velocity_fields is not None:
            self.velocity_fields = _validate_ndarray(initial_velocity_fields, required_shape=(*self.template.shape, self.num_timesteps, self.template.ndim))
        else:
            self.velocity_fields = np.zeros((*self.template.shape, self.num_timesteps, self.template.ndim))
        self.phi = np.copy(self.template_coords)
        self.affine_phi = np.copy(self.template_coords)
        self.phi_inv = np.copy(self.template_coords)
        self.phi_inv_affine_inv = np.copy(self.target_coords)
        self.fourier_velocity_fields = np.zeros_like(self.velocity_fields, np.complex128)
        self.matching_weights = np.ones_like(self.target)
        self.deformed_template_to_time = []
        self.deformed_template = interpn(
            points=self.template_axes, 
            values=self.template, 
            xi=self.phi_inv_affine_inv, 
            bounds_error=False, 
            fill_value=None, 
        )
        if spatially_varying_contrast_map:
            if initial_contrast_coefficients is None:
                self.contrast_coefficients = np.zeros((*self.target.shape, self.contrast_order + 1))
            else:
                self.contrast_coefficients = _validate_ndarray(initial_contrast_coefficients, broadcast_to_shape=(*self.target.shape, self.contrast_order + 1))
        else:
            if initial_contrast_coefficients is None:
                self.contrast_coefficients = np.zeros(self.contrast_order + 1)
            else:
                self.contrast_coefficients = _validate_ndarray(initial_contrast_coefficients, required_shape=(self.contrast_order + 1))
        self.contrast_coefficients[..., 0] = np.mean(self.target) - np.mean(self.template) * np.std(self.target) / np.std(self.template)
        if self.contrast_order > 0: self.contrast_coefficients[..., 1] = np.std(self.target) / np.std(self.template)
        self.contrast_polynomial_basis = np.empty((*self.target.shape, self.contrast_order + 1))
        for power in range(self.contrast_order + 1):
            self.contrast_polynomial_basis[..., power] = self.deformed_template**power
        self.contrast_deformed_template = np.sum(self.contrast_polynomial_basis * self.contrast_coefficients, axis=-1) # Initialized value not used.

        # Accumulators.
        self.matching_energies = []
        self.regularization_energies = []
        self.total_energies = []
        # For optional calibration plots.
        if self.calibrate:
            self.affines = []
            self.maximum_velocities = [0] * self.num_affine_only_iterations

        # Preempt known error.
        if np.any(np.array(self.template.shape) == 1) or np.any(np.array(self.target.shape) == 1):
            raise RuntimeError(f"Known issue: Images with a 1 in their shape are not supported by scipy.interpolate.interpn.\n"
                               f"self.template.shape: {self.template.shape}, self.target.shape: {self.target.shape}.\n")

    def register(self):
        """
        Register the template to the target using the current state of the attributes.
        
        Return a dictionary of relevant quantities most notably including the transformations:
            phi_inv_affine_inv is the position field that maps the template to the target.
            affine_phi is the position field that maps the target to the template.
        """

        # Iteratively perform each step of the registration.
        for iteration in range(self.num_iterations):
            # If self.track_progress_every_n > 0, print progress updates every 10 iterations.
            if self.track_progress_every_n > 0 and not iteration % self.track_progress_every_n:
                print(
                    f"Progress: iteration {iteration}/{self.num_iterations}"
                    f"{' rigid' if iteration < self.num_rigid_affine_iterations else ''}"
                    f"{' affine only' if iteration < self.num_affine_only_iterations else ' affine and deformative'}."
                )

            # Forward pass: apply transforms to the template and compute the costs.

            # Compute position_field from velocity_fields.
            self._update_and_apply_position_field()
            # Contrast transform the deformed_template.
            self._apply_contrast_map()
            # Compute weights. 
            # This is the expectation step of the expectation maximization algorithm.
            if self.classify_and_weight_voxels and iteration % 1 == 0: self._compute_weights()
            # Compute cost.
            self._compute_cost()

            # Backward pass: update contrast map, affine, & velocity_fields.
            
            # Compute contrast map.
            self._compute_contrast_map()

            # Compute affine gradient.
            affine_inv_gradient = self._compute_affine_inv_gradient()
            # Compute velocity_fields gradient.
            if iteration >= self.num_affine_only_iterations: velocity_fields_gradients = self._compute_velocity_fields_gradients()
            # Update affine.
            self._update_affine(affine_inv_gradient, iteration)  # rigid_only=iteration < self.rigid_only_iterations)
            # Update velocity_fields.
            if iteration >= self.num_affine_only_iterations: self._update_velocity_fields(velocity_fields_gradients)
        # End for loop.

        # Compute affine_phi in case there were only affine-only iterations.
        self._compute_affine_phi()

        # Optionally display useful plots for calibrating the registration parameters.
        if self.calibrate:
            self._generate_calibration_plots()
        
        # Note: the user-level lddmm_transform_image function relies on many of these specific outputs with these specific keys to function. 
        # ----> Check the lddmm_transform_image function signature before adjusting these outputs.
        return dict(
            # Core.
            affine=self.affine,
            phi=self.phi,
            phi_inv=self.phi_inv,
            affine_phi=self.affine_phi,
            phi_inv_affine_inv=self.phi_inv_affine_inv,
            contrast_coefficients=self.contrast_coefficients,
            velocity_fields=self.velocity_fields,

            # Helpers.
            template_resolution=self.template_resolution,
            target_resolution=self.target_resolution,

            # Accumulators.
            matching_energies=self.matching_energies,
            regularization_energies=self.regularization_energies,
            total_energies=self.total_energies,

            # Debuggers.
            lddmm=self,
        )
        # TODO:
        '''
        a new take on the return 'value':

        affine_phi, phi_inv_affine_inv, position_field_components, diagnostics (everything else)
        
        position_field --> map_coordinates coords: subtract coordinate of the first pixel and divide by pixel size in each dimension
        '''
        # return dict(**params) --> transform_necessary_values, just_cuz_values, calibration_accumulators

    def _update_and_apply_position_field(self):
        """
        Calculate phi_inv from v
        Compose on the right with Ainv
        Apply phi_inv_affine_inv to template

        Accesses attributes:
            template
            template_axes
            template_coords
            target_coords
            num_timesteps
            delta_t
            affine
            velocity_fields
            phi_inv
            phi_inv_affine_inv
            deformed_template_to_time
            deformed_template

        Updates attributes:
            phi_inv
            phi_inv_affine_inv
            deformed_template_to_time
            deformed_template
        """

        # Set self.phi_inv to identity.
        self.phi_inv = np.copy(self.template_coords)
        
        # Reset self.deformed_template_to_time.
        self.deformed_template_to_time = []
        for timestep in range(self.num_timesteps):
            # Compute phi_inv.
            sample_coords = self.template_coords - self.velocity_fields[..., timestep, :] * self.delta_t
            self.phi_inv = interpn(
                points=self.template_axes, 
                values=self.phi_inv - self.template_coords, 
                xi=sample_coords, 
                bounds_error=False, 
                fill_value=None, 
            ) + sample_coords

            # Compute deformed_template_to_time
            self.deformed_template_to_time.append(
                interpn(
                    points=self.template_axes, 
                    values=self.template, 
                    xi=self.phi_inv, 
                    bounds_error=False, 
                    fill_value=None, 
                )
            )
            
            # End time loop.

        # Apply affine_inv to target_coords by multiplication.
        affine_inv_target_coords = _multiply_coords_by_affine(inv(self.affine), self.target_coords)

        # Apply phi_inv to affine_inv_target_coords.
        self.phi_inv_affine_inv = interpn(
            points=self.template_axes, 
            values=self.phi_inv - self.template_coords, 
            xi=affine_inv_target_coords, 
            bounds_error=False, 
            fill_value=None, 
        ) + affine_inv_target_coords

        # Apply phi_inv_affine_inv to template.
        # deformed_template is sampled at the coordinates of the target.
        self.deformed_template = interpn(
            points=self.template_axes, 
            values=self.template, 
            xi=self.phi_inv_affine_inv, 
            bounds_error=False, 
            fill_value=None, 
        )


    def _apply_contrast_map(self):
        """
        Apply contrast_coefficients to deformed_template to produce contrast_deformed_template.

        Accsses attributes:
            contrast_polynomial_basis
            contrast_coefficients
            contrast_deformed_template


        Updates attributes:
            contrast_deformed_template
        """

        self.contrast_deformed_template = np.sum(self.contrast_polynomial_basis * self.contrast_coefficients, axis=-1)


    def _compute_weights(self):
        """
        Compute the matching_weights between the contrast_deformed_template and the target.

        Accsses attributes:
            target
            sigma_matching
            sigma_artifact
            sigma_background
            artifact_prior
            background_prior
            contrast_deformed_template
            artifact_mean_value
            background_mean_value
            matching_weights

        Updates attributes:
            artifact_mean_value
            background_mean_value
            matching_weights
        """
        
        likelihood_matching = np.exp((self.contrast_deformed_template - self.target)**2 * (-1/(2 * self.sigma_matching**2))) / np.sqrt(2 * np.pi * self.sigma_matching**2)
        likelihood_artifact = np.exp((self.artifact_mean_value - self.target)**2 * (-1/(2 * self.sigma_artifact**2))) / np.sqrt(2 * np.pi * self.sigma_artifact**2)
        likelihood_background = np.exp((self.background_mean_value - self.target)**2 * (-1/(2 * self.sigma_background**2))) / np.sqrt(2 * np.pi * self.sigma_background**2)

        # Account for priors.
        likelihood_matching *= 1 - self.artifact_prior - self.background_prior
        likelihood_artifact *= self.artifact_prior
        likelihood_background *= self.background_prior

        # Where the denominator is less than 1e-6 of its maximum, set it to 1e-6 of its maximum to avoid division by zero.
        likelihood_sum = likelihood_matching + likelihood_artifact + likelihood_background
        likelihood_sum_max = np.max(likelihood_sum)
        likelihood_sum[likelihood_sum < 1e-6 * likelihood_sum_max] = 1e-6 * likelihood_sum_max

        self.matching_weights = likelihood_matching / likelihood_sum
        artifact_weights = likelihood_artifact / likelihood_sum
        background_weights = likelihood_background / likelihood_sum

        self.artifact_mean_value = np.mean(self.target * artifact_weights) / np.mean(artifact_weights)
        self.background_mean_value = np.mean(self.target * background_weights) / np.mean(background_weights)


    def _compute_cost(self):
        """
        Compute the matching cost using a weighted sum of square error.

        Accsses attributes:
            target
            template
            template_resolution
            target_resolution
            contrast_deformed_template
            sigma_regularization
            sigma_matching
            delta_t
            fourier_high_pass_filter
            fourier_velocity_fields
            matching_weights
            matching_energies
            regularization_energies
            total_energies

        Updates attributes:
            matchin_energies
            regularization_energies
            total_energies
        """

        matching_energy = (
            np.sum((self.contrast_deformed_template - self.target)**2 * self.matching_weights) * 
            1/(2 * self.sigma_matching**2) * np.prod(self.target_resolution)
        )

        regularization_energy = (
            np.sum(np.sum(np.abs(self.fourier_velocity_fields)**2, axis=(-1,-2)) * self.fourier_high_pass_filter**2) * 
            (np.prod(self.template_resolution) * self.delta_t / (2 * self.sigma_regularization**2) / self.template.size)
        )

        total_energy = matching_energy + regularization_energy

        # Accumulate energies.
        self.matching_energies.append(matching_energy)
        self.regularization_energies.append(regularization_energy)
        self.total_energies.append(total_energy)


    def _compute_contrast_map(self):
            """
            Compute contrast_coefficients mapping deformed_template to target.

            Accesses attributes:
                target
                target_resolution
                deformed_template
                spatially_varying_contrast_map
                sigma_matching
                contrast_order
                sigma_contrast
                contrast_tolerance
                contrast_maxiter
                matching_weights
                contrast_polynomial_basis
                contrast_coefficients

            Updates attributes:
                contrast_polynomial_basis
                contrast_coefficients
            """

            # Update self.contrast_polynomial_basis.
            for power in range(self.contrast_order + 1):
                self.contrast_polynomial_basis[..., power] = self.deformed_template**power

            if self.spatially_varying_contrast_map:
                # Compute and set self.contrast_coefficients for self.spatially_varying_contrast_map == True.

                '=============================================================================================================================='

                # C is contrast_coefficients.
                # B is the contrast_polynomial_basis.
                # W is weights.
                # L is contrast_high_pass_filter.
                # This is the minimization problem: sum(|BC - J|^2 W^2 / 2) + sum(|LC|^2 / 2).
                # The linear equation we need to solve for C is this: W^2 B^T B C  + L^T L C = W^2 B^T J.
                # Where W acts by pointwise multiplication, B acts by matrix multiplication at every point, and L acts by filtering in the Fourier domain.
                # Let L C = D. --> C = L^{-1} D.
                # This reformulates the problem to: W^2 B^T B L^{-1} D + L^T D = W^2 B^T J.
                # Then, to make it nicer we act on both sides with L^{-T}, yielding: L^{-T}(B^T B) L^{-1}D + D = L^{-T} W^2 B^t J.
                # Then we factor the left side: [L^{-T} B^T  B L^{-1} + identity]D = L^{-T}W^2 B^T J

                spatial_ndim = self.contrast_polynomial_basis.ndim - 1

                # Represents W in the equation.
                weights = np.sqrt(self.matching_weights) / self.sigma_matching

                # Represents the right hand side of the equation.
                right_hand_side = self.contrast_polynomial_basis * (weights**2 * self.target)[..., None]

                # Reformulate with block elimination.
                high_pass_contrast_coefficients = np.fft.ifftn(np.fft.fftn(self.contrast_coefficients, axes=range(spatial_ndim)) * self.contrast_high_pass_filter[..., None], axes=range(spatial_ndim)).real
                low_pass_right_hand_side = np.fft.ifftn(np.fft.fftn(right_hand_side, axes=range(spatial_ndim)) / self.contrast_high_pass_filter[..., None], axes=range(spatial_ndim)).real
                for _ in range(self.contrast_maxiter):
                    linear_operator_high_pass_contrast_coefficients = np.fft.ifftn(np.fft.fftn((
                        np.sum(
                            np.fft.ifftn(np.fft.fftn(high_pass_contrast_coefficients, axes=range(spatial_ndim)) / self.contrast_high_pass_filter[..., None], axes=range(spatial_ndim)).real * self.contrast_polynomial_basis, 
                            axis=-1,
                        ) * weights**2
                    )[..., None] * self.contrast_polynomial_basis, axes=range(spatial_ndim)) / self.contrast_high_pass_filter[..., None], axes=range(spatial_ndim)).real + high_pass_contrast_coefficients
                    residual = linear_operator_high_pass_contrast_coefficients - low_pass_right_hand_side
                    # Compute the optimal step size.
                    linear_operator_residual = np.fft.ifftn(np.fft.fftn((
                        np.sum(
                            np.fft.ifftn(np.fft.fftn(residual, axes=range(spatial_ndim)) / self.contrast_high_pass_filter[..., None], axes=range(spatial_ndim)).real * self.contrast_polynomial_basis, 
                            axis=-1,
                        ) * weights**2
                    )[..., None] * self.contrast_polynomial_basis, axes=range(spatial_ndim)) / self.contrast_high_pass_filter[..., None], axes=range(spatial_ndim)).real + residual
                    optimal_stepsize = np.sum(residual**2) / np.sum(linear_operator_residual * residual)
                    # Take gradient descent step at half the optimal step size.
                    high_pass_contrast_coefficients -= optimal_stepsize * residual / 2
                
                self.contrast_coefficients = np.fft.ifftn(np.fft.fftn(high_pass_contrast_coefficients, axes=range(spatial_ndim)) / self.contrast_high_pass_filter[..., None], axes=range(spatial_ndim)).real



                APPLYOP_to_contrast_coefficients = np.fft.ifftn(np.fft.fftn((
                    np.sum(
                        np.fft.ifftn(np.fft.fftn(self.contrast_coefficients, axes=range(spatial_ndim)) / self.contrast_high_pass_filter[..., None], axes=range(spatial_ndim)).real * self.contrast_polynomial_basis, 
                        axis=-1,
                    ) * weights**2
                )[..., None] * self.contrast_polynomial_basis, axes=range(spatial_ndim)) / self.contrast_high_pass_filter[..., None], axes=range(spatial_ndim)).real + self.contrast_coefficients






                '=============================================================================================================================='

            # Spatially varying contrast code regularized like velocity_fields using scipy.sparse.linalg.cg, pending depracation in favor of above.
            elif False:
                # Compute and set self.contrast_coefficients for self.spatially_varying_contrast_map == True.

                # Shape: (*self.target.shape, self.contrast_order + 1, self.contrast_order + 1)
                contrast_polynomial_basis_transpose = np.transpose(self.contrast_polynomial_basis[..., None], (*range(self.target.ndim), self.target.ndim + 1, self.target.ndim))
                matching_matrix = (self.contrast_polynomial_basis[..., None] * self.matching_weights[..., None, None] / self.sigma_matching**2) @ contrast_polynomial_basis_transpose

                # Shape: (*self.target.shape, self.contrast_order + 1)
                target_matrix = self.target[..., None] * self.contrast_polynomial_basis * self.matching_weights[..., None] / self.sigma_matching**2

                def _matvec(contrast_coefficients, self=self, matching_matrix=matching_matrix):
                    """Returns (matching_matrix @ contrast_coefficients.reshape(self.contrast_coefficients.shape) + regularization_matrix).ravel()."""

                    contrast_coefficients = contrast_coefficients.reshape(self.contrast_coefficients.shape)

                    # regularization_matrix = ((identity - contrast_smooth_length**2 * Laplacian)**(2 * fourier_filter_power) / sigma_contrast**2) @ contrast_coefficients
                    regularization_matrix = np.copy(contrast_coefficients)
                    for _ in range(2 * self.fourier_filter_power):
                        regularization_matrix_laplacian = np.zeros_like(regularization_matrix)
                        for dim in range(self.target.ndim):
                            regularization_matrix_laplacian += (np.roll(regularization_matrix, 1, axis=dim) - 2 * regularization_matrix + np.roll(regularization_matrix, -1, axis=dim)) / self.target_resolution[dim]**2
                        # regularization_matrix_laplacian is now the Laplacian of regularization_matrix.
                        regularization_matrix -= regularization_matrix_laplacian * self.contrast_smooth_length**2
                    regularization_matrix /= self.sigma_contrast**2

                    return (matching_matrix @ contrast_coefficients[..., None] + regularization_matrix[..., None]).ravel()
                linear_operator = LinearOperator((self.contrast_coefficients.size, self.contrast_coefficients.size), matvec=_matvec)

                # Use scipy.sparse.linalg.cg to update self.contrast_coefficients.
                contrast_coefficients_update = cg(linear_operator, target_matrix.ravel(), x0=self.contrast_coefficients.ravel(), tol=self.contrast_tolerance, maxiter=self.contrast_maxiter)
                if contrast_coefficients_update[1] != 0:
                    warnings.warn(
                        f"scipy.sparse.linalg.cg in _compute_contrast_map has not successfully converged with convergence code {contrast_coefficients_update[1]}.", 
                        RuntimeWarning
                    )
                self.contrast_coefficients = contrast_coefficients_update[0].reshape(self.contrast_coefficients.shape)

            # Penalty on derivative squared, pending depracation in favor of above.
            elif False:
                # Compute and set self.contrast_coefficients for self.spatially_varying_contrast_map == True.

                # Shape: (*self.target.shape, self.contrast_order + 1, self.contrast_order + 1)
                contrast_polynomial_basis_transpose = np.transpose(self.contrast_polynomial_basis[..., None], (*range(self.target.ndim), self.target.ndim + 1, self.target.ndim))
                matching_matrix = (self.contrast_polynomial_basis[..., None] * self.matching_weights[..., None, None] / self.sigma_matching**2) @ contrast_polynomial_basis_transpose

                # Shape: (*self.target.shape, self.contrast_order + 1)
                target_matrix = self.target[..., None] * self.contrast_polynomial_basis * self.matching_weights[..., None] / self.sigma_matching**2

                def _matvec(contrast_coefficients, self=self, matching_matrix=matching_matrix):
                    """Returns (matching_matrix @ contrast_coefficients.reshape(self.contrast_coefficients.shape) - regularization_matrix).ravel()."""

                    contrast_coefficients = contrast_coefficients.reshape(self.contrast_coefficients.shape)

                    regularization_matrix = np.zeros_like(contrast_coefficients)
                    for dim in range(self.target.ndim):
                        regularization_matrix += (np.roll(contrast_coefficients, 1, axis=dim) - 2 * contrast_coefficients + np.roll(contrast_coefficients, -1, axis=dim)) / self.target_resolution[dim]**2
                    # regularization_matrix is now the Laplacian of contrast_coefficients.
                    regularization_matrix /= self.sigma_contrast**2

                    return (matching_matrix @ contrast_coefficients[..., None] - regularization_matrix[..., None]).ravel()
                linear_operator = LinearOperator((self.contrast_coefficients.size, self.contrast_coefficients.size), matvec=_matvec)

                # Use scipy.sparse.linalg.cg to update self.contrast_coefficients.
                contrast_coefficients_update = cg(linear_operator, target_matrix.ravel(), x0=self.contrast_coefficients.ravel(), tol=self.contrast_tolerance, maxiter=self.contrast_maxiter)
                if contrast_coefficients_update[1] != 0:
                    warnings.warn(
                        f"scipy.sparse.linalg.cg in _compute_contrast_map has not successfully converged with convergence code {contrast_coefficients_update[1]}.", 
                        RuntimeWarning
                    )
                self.contrast_coefficients = contrast_coefficients_update[0].reshape(self.contrast_coefficients.shape)

            else:
                # Compute and set self.contrast_coefficients for self.spatially_varying_contrast_map == False.

                # Ravel necessary components for convenient matrix multiplication.
                deformed_template_ravel = np.ravel(self.deformed_template)
                target_ravel = np.ravel(self.target)
                matching_weights_ravel = np.ravel(self.matching_weights)
                contrast_polynomial_basis_semi_ravel = self.contrast_polynomial_basis.reshape(self.target.size, -1) # A view, not a copy.

                # Create intermediate composites.
                basis_transpose_basis = np.matmul(contrast_polynomial_basis_semi_ravel.T * matching_weights_ravel, contrast_polynomial_basis_semi_ravel)
                basis_transpose_target = np.matmul(contrast_polynomial_basis_semi_ravel.T * matching_weights_ravel, target_ravel)

                # Solve for contrast_coefficients.
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', message='Ill-conditioned matrix')
                    try:
                        self.contrast_coefficients = solve(basis_transpose_basis, basis_transpose_target, assume_a='pos')
                    except np.linalg.LinAlgError as e:
                        raise np.linalg.LinAlgError(f"This exception may have been raised because the contrast_polynomial_basis vectors were not independent, i.e. the template is constant.") from e


    def _compute_affine_inv_gradient(self):
        """
        Compute and return the affine_inv gradient.

        Accesss attributes:
            template
            target
            template_resolution
            template_axes
            target_coords
            deformed_template
            contrast_deformed_template
            sigma_matching
            contrast_order
            phi_inv
            matching_weights
            contrast_coefficients
            affine

        Updates attributes:
            None

        Returns:
            affine_inv_gradient
        """

        # Generate the template image deformed by phi_inv but not affected by the affine.
        non_affine_deformed_template = interpn(
            points=self.template_axes, 
            values=self.template, 
            xi=self.phi_inv, 
            bounds_error=False, 
            fill_value=None, 
        )

        # Compute the gradient of non_affine_deformed_template.
        non_affine_deformed_template_gradient = np.stack(np.gradient(non_affine_deformed_template, *self.template_resolution), -1)

        # Apply the affine to each component of non_affine_deformed_template_gradient.
        sample_coords = _multiply_coords_by_affine(inv(self.affine), self.target_coords)
        deformed_template_gradient = interpn(
            points=self.template_axes,
            values=non_affine_deformed_template_gradient,
            xi=sample_coords,
            bounds_error=False,
            fill_value=None,
        )

        # Reshape and broadcast deformed_template_gradient from shape (x,y,z,3) to (x,y,z,3,1) to (x,y,z,3,4) - for a 3D example.
        deformed_template_gradient_broadcast = np.repeat(np.expand_dims(deformed_template_gradient, -1), repeats=self.target.ndim + 1, axis=-1)

        # Construct homogenous_target_coords by appending 1's at the end of the last dimension throughout self.target_coords.
        ones = np.ones((*self.target.shape, 1))
        homogenous_target_coords = np.concatenate((self.target_coords, ones), -1)
        
        # For a 3D example:

        # deformed_template_gradient_broadcast has shape (x,y,z,3,4).
        # homogenous_target_coords has shape (x,y,z,4).

        # To repeat homogenous_target_coords along the 2nd-last dimension of deformed_template_gradient_broadcast, 
        # we reshape homogenous_target_coords from shape (x,y,z,4) to shape (x,y,z,1,4) and let that broadcast to shape (x,y,z,3,4).

        matching_affine_inv_gradient = deformed_template_gradient_broadcast * np.expand_dims(homogenous_target_coords, -2)

        # Get error term.
        matching_error_prime = (self.contrast_deformed_template - self.target) * self.matching_weights / self.sigma_matching**2
        contrast_map_prime = np.zeros_like(self.target, float)
        for power in range(1, self.contrast_order + 1):
            contrast_map_prime += power * self.deformed_template**(power - 1) * self.contrast_coefficients[..., power]
        d_matching_d_deformed_template = matching_error_prime * contrast_map_prime

        affine_inv_gradient = matching_affine_inv_gradient * d_matching_d_deformed_template[...,None,None]

        # Note: before implementing Gauss Newton below, affine_inv_gradient_reduction as defined below was the previous returned value for the affine_inv_gradient.
        # For 3D case, this has shape (3,4).
        affine_inv_gradient_reduction = np.sum(affine_inv_gradient, tuple(range(self.target.ndim)))

        # Reshape to a single vector. For a 3D case this becomes shape (12,).
        affine_inv_gradient_reduction = affine_inv_gradient_reduction.ravel()

        # For a 3D case, matching_affine_inv_gradient has shape (x,y,z,3,4).
        # For a 3D case, affine_inv_hessian_approx is matching_affine_inv_gradient reshaped to shape (x,y,z,12,1), 
        # then matrix multiplied by itself transposed on the last two dimensions, then summed over the spatial dimensions
        # to resultant shape (12,12).
        affine_inv_hessian_approx = matching_affine_inv_gradient * ((contrast_map_prime * np.sqrt(self.matching_weights) / self.sigma_matching)[...,None,None])
        affine_inv_hessian_approx = affine_inv_hessian_approx.reshape(*matching_affine_inv_gradient.shape[:-2], -1, 1)
        affine_inv_hessian_approx = affine_inv_hessian_approx @ affine_inv_hessian_approx.reshape(*affine_inv_hessian_approx.shape[:-2], 1, -1)
        affine_inv_hessian_approx = np.sum(affine_inv_hessian_approx, tuple(range(self.target.ndim)))

        # Solve for affine_inv_gradient.
        try:
            affine_inv_gradient = solve(affine_inv_hessian_approx, affine_inv_gradient_reduction, assume_a='pos').reshape(matching_affine_inv_gradient.shape[-2:])
        except np.linalg.LinAlgError as exception:
            raise RuntimeError(
                "The Hessian was not invertible in the Gauss-Newton update of the affine transform. "
                "This may be because the image was constant along one or more dimensions. "
                "Consider removing any constant dimensions. "
                "Otherwise you may try using a smaller value for affine_stepsize, a smaller value for deformative_stepsize, or a larger value for sigma_regularization. "
                "The calibrate=True option may be of use in determining optimal parameter values."
            ) from exception
        # Append a row of zeros at the end of the 0th dimension.
        zeros = np.zeros((1, self.target.ndim + 1))
        affine_inv_gradient = np.concatenate((affine_inv_gradient, zeros), 0)

        return affine_inv_gradient


    def _update_affine(self, affine_inv_gradient, iteration):
        """
        Update self.affine based on affine_inv_gradient.

        If iteration < self.num_rigid_affine_iterations, project self.affine to a rigid affine.

        If self.
         is provided, it is imposed on self.affine.

        if self.calibrate, appends the current self.affine to self.affines.

        Accesses attributes:
            calibrate
            fixed_affine_scale
            affine_stepsize
            affine
            affines

        Updates attributes:
            affine
            affines
        """
        
        affine_inv = inv(self.affine)

        affine_inv -= affine_inv_gradient * self.affine_stepsize

        self.affine = inv(affine_inv)

        # Set scale of self.affine if appropriate.
        if self.fixed_affine_scale is not None:
            U, _, Vh = svd(self.affine[:-1, :-1])
            self.affine[:-1, :-1] = U @ np.diag([self.fixed_affine_scale] * (len(self.affine) - 1)) @ Vh
        # If self.fixed_affine_scale was not provided (is None), project self.affine to a rigid affine if appropriate.
        elif iteration < self.num_rigid_affine_iterations:
            U, _, Vh = svd(self.affine[:-1, :-1])
            self.affine[:-1, :-1] = U @ Vh

        # Save affine for calibration plotting.
        if self.calibrate:
            self.affines.append(self.affine)


    def _compute_velocity_fields_gradients(self):
        """
        Compute and return the gradients of the self.velocity_fields.

        Accesses attributes:
            template
            target
            template_axes
            target_axes
            template_coords
            template_resolution
            deformed_template_to_time
            deformed_template
            contrast_deformed_template
            sigma_regularization
            sigma_matching
            contrast_order
            num_timesteps
            delta_t
            low_pass_filter
            preconditioner_low_pass_filter
            matching_weights
            contrast_coefficients
            velocity_fields
            affine
            phi
            affine_phi

        Updates attributes:
            phi
            affine_phi
        """

        matching_error_prime = (self.contrast_deformed_template - self.target) * self.matching_weights / self.sigma_matching**2
        contrast_map_prime = np.zeros_like(self.target, float)
        for power in range(1, self.contrast_order + 1):
            contrast_map_prime += power * self.deformed_template**(power - 1) * self.contrast_coefficients[..., power]
        d_matching_d_deformed_template = matching_error_prime * contrast_map_prime
        d_matching_d_deformed_template_padded = np.pad(d_matching_d_deformed_template, 2, mode='constant', constant_values=0)

        # Set self.phi to identity. self.phi is secretly phi_1t_inv but at the end of the loop 
        # it will be phi_10_inv = phi_01 = phi.
        self.phi = np.copy(self.template_coords)

        # Loop backwards across time.
        d_matching_d_velocities = []
        for timestep in reversed(range(self.num_timesteps)):

            # Update phi.
            sample_coords = self.template_coords + self.velocity_fields[..., timestep, :] * self.delta_t
            self.phi = interpn(
                points=self.template_axes, 
                values=self.phi - self.template_coords, 
                xi=sample_coords, 
                bounds_error=False, 
                fill_value=None, 
            ) + sample_coords

            # Apply affine by multiplication.
            # This transforms error in the target space back to time t.
            self.affine_phi = _multiply_coords_by_affine(self.affine, self.phi)

            # Compute the determinant of the gradient of self.phi.
            grad_phi = np.stack(np.gradient(self.phi, *self.template_resolution, axis=tuple(range(self.template.ndim))), -1)
            det_grad_phi = _compute_tail_determinant(grad_phi)

            # Transform error in target space back to time t.
            error_at_t = interpn(
                points=_compute_axes(d_matching_d_deformed_template_padded.shape, self.target_resolution),
                values=d_matching_d_deformed_template_padded,
                xi=self.affine_phi,
                bounds_error=False,
                fill_value=None,
            )

            # The gradient of the template image deformed to time t.
            deformed_template_to_time_gradient = np.stack(np.gradient(self.deformed_template_to_time[timestep], *self.template_resolution, axis=tuple(range(self.template.ndim))), -1)

            # The derivative of the matching cost with respect to the velocity at time t
            # is the product of 
            # (the error deformed to time t), 
            # (the template gradient deformed to time t), 
            # & (the determinant of the jacobian of the transformation).
            d_matching_d_velocity_at_t = np.expand_dims(error_at_t * det_grad_phi, -1) * deformed_template_to_time_gradient * (-1.0) * det(self.affine)

            # To convert from derivative to gradient we smooth by applying a physical-unit low-pass filter in the frequency domain.
            matching_cost_at_t_gradient = np.fft.fftn(d_matching_d_velocity_at_t, axes=tuple(range(self.template.ndim))) * np.expand_dims(self.low_pass_filter, -1)
            # Add the gradient of the regularization term.
            matching_cost_at_t_gradient += np.fft.fftn(self.velocity_fields[...,timestep,:], axes=tuple(range(self.template.ndim))) / self.sigma_regularization**2
            # Multiply by a voxel-unit low-pass filter to further smooth.
            matching_cost_at_t_gradient *= np.expand_dims(self.preconditioner_low_pass_filter, -1)
            # Invert fourier transform back to the spatial domain.
            d_matching_d_velocity_at_t = np.fft.ifftn(matching_cost_at_t_gradient, axes=tuple(range(self.template.ndim))).real

            d_matching_d_velocities.insert(0, d_matching_d_velocity_at_t)

        return d_matching_d_velocities

    
    def _update_velocity_fields(self, velocity_fields_gradients):
        """
        Update self.velocity_fields based on velocity_fields_gradient.

        if self.calibrate, calculates and appends the maximum velocity to self.maximum_velocities.

        Accesses attributes:
            calibrate
            deformative_stepsize
            num_timesteps
            velocity_fields
            maximum_velocities

        Updates attributes:
            velocity_fields
            maximum_velocities
        """


        for timestep in range(self.num_timesteps):
            velocity_fields_update = velocity_fields_gradients[timestep] * self.deformative_stepsize
            # Apply a sigmoid squashing function to the velocity_fields_update to ensure they yield an update of less than self.maximum_velocity_fields_update voxels while remaining smooth.
            velocity_fields_update_norm = np.sqrt(np.sum(velocity_fields_update**2, axis=-1))
            velocity_fields_update = (
                velocity_fields_update / velocity_fields_update_norm[..., None] * 
                np.arctan(velocity_fields_update_norm[..., None] * np.pi / 2 / self.maximum_velocity_fields_update) * 
                self.maximum_velocity_fields_update * 2 / np.pi
            )

            self.velocity_fields[...,timestep,:] -= velocity_fields_update
            
        # Save maximum velocity for calibration plotting.
        if self.calibrate:
            maximum_velocity = np.sqrt(np.sum(self.velocity_fields**2, axis=-1)).max()
            self.maximum_velocities.append(maximum_velocity)

    
    def _compute_affine_phi(self):
        """
        Compute and set self.affine_phi. Called once in case there were no deformative iterations to set it.

        Accesses attributes:
            template_axes
            template_coords
            delta_t
            num_timesteps
            velocity_fields
            affine
            phi
            affine_phi

        Updates attributes:
            phi
            affine_phi
        """

        # Set self.phi to identity. self.phi is secretly phi_1t_inv but at the end of the loop 
        # it will be phi_10_inv = phi_01 = phi.
        self.phi = np.copy(self.template_coords)

        # Loop backwards across time.
        for timestep in reversed(range(self.num_timesteps)):

            # Update phi.
            sample_coords = self.template_coords + self.velocity_fields[..., timestep, :] * self.delta_t
            self.phi = interpn(
                points=self.template_axes, 
                values=self.phi - self.template_coords, 
                xi=sample_coords, 
                bounds_error=False, 
                fill_value=None, 
            ) + sample_coords

            # Apply affine by multiplication.
            # This transforms error in the target space back to time t.
            self.affine_phi = _multiply_coords_by_affine(self.affine, self.phi)
    

    # TODO: move into example file.
    def _generate_calibration_plots(self):
        """
        Plot the energies, maximum velocities, translation components, and linear components as functions of the number of iterations.
        """

        fig, axes = plt.subplots(2, 2, figsize=(6, 6))

        # Plot matching, regularization, and total energies.
        ax = axes[0, 0]
        ax.plot(list(zip(self.matching_energies, self.regularization_energies, self.total_energies)))
        ax.set_title('Energies')

        # Plot the maximum velocity.
        ax = axes[0, 1]
        ax.plot(self.maximum_velocities)
        ax.set_title('Maximum\nvelocity')

        # Plot affine[:, :-1], the translation components.
        translations = [affine[:-1, -1] for affine in self.affines]
        ax = axes[1, 0]
        ax.plot(translations)
        ax.set_title('Translation\ncomponents')

        # Plot self.affine[:-1, :-1], the linear transformation components.
        linear_components = [affine[:-1, :-1].ravel() for affine in self.affines]
        ax = axes[1, 1]
        ax.plot(linear_components)
        ax.set_title('Linear\ncomponents')

    # End _Lddmm.

r'''
  _    _                          __                          _     _                       
 | |  | |                        / _|                        | |   (_)                      
 | |  | |  ___    ___   _ __    | |_   _   _   _ __     ___  | |_   _    ___    _ __    ___ 
 | |  | | / __|  / _ \ | '__|   |  _| | | | | | '_ \   / __| | __| | |  / _ \  | '_ \  / __|
 | |__| | \__ \ |  __/ | |      | |   | |_| | | | | | | (__  | |_  | | | (_) | | | | | \__ \
  \____/  |___/  \___| |_|      |_|    \__,_| |_| |_|  \___|  \__| |_|  \___/  |_| |_| |___/
                                                                                            
'''

def lddmm_register(
    # Images.
    template,
    target,
    # Image resolutions.
    template_resolution=None,
    target_resolution=None,
    # Multiscale.
    multiscales=None,
    # Iterations.
    num_iterations=None,
    num_affine_only_iterations=None,
    num_rigid_affine_iterations=None,
    # Stepsizes.
    affine_stepsize=None,
    deformative_stepsize=None,
    # Affine specifiers.
    fixed_affine_scale=None,
    # Velocity field specifiers.
    sigma_regularization=None,
    velocity_smooth_length=None,
    preconditioner_velocity_smooth_length=None,
    maximum_velocity_fields_update=None,
    num_timesteps=None,
    # Contrast map specifiers.
    contrast_order=None,
    spatially_varying_contrast_map=None,
    contrast_maxiter=None,
    contrast_tolerance=None,
    sigma_contrast=None,
    contrast_smooth_length=None,
    # Smoothness vs. accuracy tradeoff.
    sigma_matching=None,
    # Classification specifiers.
    classify_and_weight_voxels=None,
    sigma_artifact=None,
    sigma_background=None,
    artifact_prior=None,
    background_prior=None,
    # Initial values.
    initial_affine=None,
    initial_contrast_coefficients=None,
    initial_velocity_fields=None,
    # Diagnostic outputs.
    calibrate=None,
    track_progress_every_n=None,
):
    """
    Compute a registration between template and target, to be applied with lddmm_transform_image.
    
    Parameters
    ----------
        template: np.ndarray
            The ideally clean template image being registered to the target.
        target: np.ndarray
            The potentially messier target image being registered to.
        template_resolution: float, seq, optional
            A scalar or list of scalars indicating the resolution of the template. Overrides 0 input. By default 1.
        target_resolution: float, seq, optional
            A scalar or list of scalars indicating the resolution of the target. Overrides 0 input. By default 1.
        multiscales: float, seq, optional
            A scalar, list of scalars, or list of lists or np.ndarray of scalars, determining the levels of downsampling at which the registration should be performed before moving on to the next. 
            Values must be either all at least 1, or all at most 1. Both options are interpreted as downsampling. For example, multiscales=[10, 3, 1] will result in the template and target being downsampled by a factor of 10 and registered. 
            This registration will be upsampled and used to initialize another registration of the template and target downsampled by 3, and then again on the undownsampled data. multiscales=[1/10, 1/3, 1] is equivalent. 
            Alternatively, the scale for each dimension can be specified, e.g. multiscales=[ [10, 5, 5], [3, 3, 3], 1] for a 3D registration will result in the template and target downsampled by [10, 5, 5], then [3, 3, 3], then [1, 1, 1]. 
            If provided with more than 1 value, all following arguments with the exceptions of initial_affine, initial_velocity_fields, and initial_contrast_coefficients, 
            which may be provided for the first value in multiscales, may optionally be provided as sequences with length equal to the number of values provided to multiscales. Each such value is used at the corresponding scale. 
            Additionally, template_resolution and target_resolution cannot be provided for each scale in multiscales. Rather, they are given once to indicate the resolution of the template and target as input.
            multiscales should be provided as descending values. By default 1.
        num_iterations: int, optional
            The total number of iterations. By default 300.
        num_affine_only_iterations: int, optional
            The number of iterations at the start of the process without deformative adjustments. By default 100.
        num_rigid_affine_iterations: int, optional
            The number of iterations at the start of the process in which the affine is kept rigid. By default 50.
        affine_stepsize: float, optional
            The unitless stepsize for affine adjustments. Should be between 0 and 1. By default 0.3.
        deformative_stepsize: float, optional
            The stepsize for deformative adjustments. Optimal values are problem-specific. If equal to 0 then the result is affine-only registration. By default 0.
        fixed_affine_scale: float, optional
            The scale to impose on the affine at all iterations. If None, no scale is imposed. Otherwise, this has the effect of making the affine always rigid. By default None.
        sigma_regularization: float, optional
            A scalar indicating the freedom to deform. Overrides 0 input. By default 10 * np.max(self.template_resolution).
        velocity_smooth_length: float, optional
            The length scale of smoothing of the velocity_fields in physical units. Determines the optimum velocity_fields smoothness. By default 2 * np.max(self.template_resolution).
        preconditioner_velocity_smooth_length: float, optional
            The length of preconditioner smoothing of the velocity_fields in physical units. Determines the optimization of the velocity_fields. By default 5 * np.max(self.template_resolution).
        maximum_velocity_fields_update: float, optional
            The maximum allowed update to the velocity_fields in units of voxels. By default 1.
        num_timesteps: int, optional
            The number of composed sub-transformations in the diffeomorphism. Overrides 0 input. By default 5.
        contrast_order: int, optional
            The order of the polynomial fit between the contrasts of the template and target. Overrides 0 input. By default 1.
        spatially_varying_contrast_map: bool, optional
            If True, uses a polynomial per voxel to compute the contrast map rather than a single polynomial. By default False.
        contrast_maxiter: int, optional
            The maximum number of iterations to converge toward the optimal contrast_coefficients if spatially_varying_contrast_map == True. Overrides 0 input. By default 5.
        contrast_tolerance: float, optional
            Deprecated. The tolerance for convergence to the optimal contrast_coefficients if spatially_varying_contrast_map == True. By default 1e-5.
        sigma_contrast: float, optional
            The scale of variation in the contrast_coefficients if spatially_varying_contrast_map == True. Overrides 0 input. By default 1.
        contrast_smooth_length: float, optional
            The length scale of smoothing of the contrast_coefficients if spatially_varying_contrast_map == True. Overrides 0 input. By default 2 * np.max(self.target_resolution).
        sigma_matching: float, optional
            An estimate of the spread of the noise in the target, 
            representing the tradeoff between the regularity and accuracy of the registration, where a smaller value should result in a less smooth, more accurate result. 
            Typically it should be set to an estimate of the standard deviation of the noise in the image, particularly with artifacts. Overrides 0 input. By default the standard deviation of the target.
        classify_and_weight_voxels: bool, optional
            If True, artifacts and background are jointly classified with registration using sigma_artifact, artifact_prior, sigma_background, and background_prior. 
            Artifacts refer to excessively bright voxels while background refers to excessively dim voxels. By default False.
        sigma_artifact: float, optional
            The level of expected variation between artifact and non-artifact intensities. Overrides 0 input. By default 5 * sigma_matching.
        sigma_background: float, optional
            The level of expected variation between background and non-background intensities. Overrides 0 input. By default 2 * sigma_matching.
        artifact_prior: float, optional
            The prior probability at which we expect to find that any given voxel is artifact. By default 1/3.
        background_prior: float, optional
            The prior probability at which we expect to find that any given voxel is background. By default 1/3.
        initial_affine: np.ndarray, optional
            The affine array that the registration will begin with. By default np.eye(template.ndim + 1).
        initial_contrast_coefficients: np.ndarray, optional
            The contrast coefficients that the registration will begin with. 
            If None, the 0th order coefficient(s) are set to np.mean(self.target) - np.mean(self.template) * np.std(self.target) / np.std(self.template), 
            if self.contrast_order > 1, the 1st order coefficient(s) are set to np.std(self.target) / np.std(self.template), 
            and all others are set to zero. By default None.
        initial_velocity_fields: np.ndarray, optional
            The velocity fields that the registration will begin with. By default all zeros.
        calibrate: bool, optional
            A boolean flag indicating whether to accumulate additional intermediate values and display informative plots for calibration purposes. By default False.
        track_progress_every_n: int, optional
            If positive, a progress update will be printed every track_progress_every_n iterations of registration. By default 0.
    
    Example:
        >>> import numpy as np
        >>> from scipy.ndimage import rotate
        >>> from skimage.registration import lddmm_register, lddmm_transform_image
        >>> # 
        >>> # Define images. The template is registered to the target image but both transformations are returned.
        >>> # template is a binary ellipse with semi-radii 5 and 8 in dimensions 0 and 1. The overall shape is (19, 25).
        >>> # target is a 30 degree rotation of template in the (1,2) plane.
        >>> # 
        >>> template = np.array([[[(col-12)**2/8**2 + (row-9)**2/5**2 <= 1 for col in range(25)] for row in range(19)]]*2, int)
        >>> target = rotate(template, 30, (1,2))
        >>> # 
        >>> # Register the template to the target, then deform the template and target to match the other.
        >>> # 
        >>> lddmm_dict = lddmm_register(template, target, deformative_stepsize = 0.5)
        >>> deformed_target = lddmm_transform_image(target, deform_to='template', **lddmm_dict)
        >>> deformed_template = lddmm_transform_image(template, deform_to='target', **lddmm_dict)

    Returns
    -------
    dict
        A dictionary containing all important saved quantities computed during the registration.

    Raises
    ------
    ValueError
        Raised if multiscales is provided with values both above and below 1.
    """

    # Validate images and resolutions.
    # Images.
    template = _validate_ndarray(template)
    target = _validate_ndarray(target, required_ndim=template.ndim)
    # Resolution.
    template_resolution = _validate_scalar_to_multi(template_resolution if template_resolution is not None else 1, template.ndim, float)
    target_resolution = _validate_scalar_to_multi(target_resolution if target_resolution is not None else 1, target.ndim, float)

    # Validate multiscales.
    # Note: this is the only argument not passed to _Lddmm.
    if multiscales is None: multiscales = 1
    try:
        multiscales = list(multiscales)
    except TypeError:
        multiscales = [multiscales]
    # multiscales is a list.
    for index, scale in enumerate(multiscales):
        multiscales[index] = _validate_scalar_to_multi(scale, size=template.ndim, dtype=float)
    multiscales = _validate_ndarray(multiscales, required_shape=(-1, template.ndim))
    # Each scale in multiscales has length template.ndim.
    if np.all(multiscales >= 1):
        multiscales = 1 / multiscales
    elif not np.all(multiscales <= 1):
        raise ValueError(f"If provided, the values in multiscales must be either all >= 1 or all <= 1.")
    # All values in multiscales are <= 1. If provided with all scales greater than or equal to 1, multiscales are ingested as their reciprocals.

    # Validate potential multiscale arguments.

    # All multiscale_lddmm_kwargs should be used in _Lddmm as scalars, not sequences.
    # Here, they are made into sequences corresponding to the length of multiscales.
    multiscale_lddmm_kwargs = dict(
        # # Images.
        # template=template,
        # target=target,
        # # Image resolutions.
        # template_resolution=template_resolution,
        # target_resolution=target_resolution,
        # Iterations.
        num_iterations=num_iterations,
        num_affine_only_iterations=num_affine_only_iterations,
        num_rigid_affine_iterations=num_rigid_affine_iterations,
        # Stepsizes.
        affine_stepsize=affine_stepsize,
        deformative_stepsize=deformative_stepsize,
        # Affine specifiers.
        fixed_affine_scale=fixed_affine_scale,
        # Velocity field specifiers.
        sigma_regularization=sigma_regularization,
        velocity_smooth_length=velocity_smooth_length,
        preconditioner_velocity_smooth_length=preconditioner_velocity_smooth_length,
        maximum_velocity_fields_update=maximum_velocity_fields_update,
        num_timesteps=num_timesteps,
        # Contrast map specifiers.
        contrast_order=contrast_order,
        spatially_varying_contrast_map=spatially_varying_contrast_map,
        contrast_maxiter=contrast_maxiter,
        contrast_tolerance=contrast_tolerance,
        sigma_contrast=sigma_contrast,
        contrast_smooth_length=contrast_smooth_length,
        # # vs. accuracy tradeoff.
        sigma_matching=sigma_matching,
        # Classification specifiers.
        classify_and_weight_voxels=classify_and_weight_voxels,
        sigma_artifact=sigma_artifact,
        sigma_background=sigma_background,
        artifact_prior=artifact_prior,
        background_prior=background_prior,
        # # Initial values.
        # initial_affine=initial_affine,
        # initial_contrast_coefficients=initial_contrast_coefficients,
        # initial_velocity_fields=initial_velocity_fields,
        # Diagnostic outputs.
        calibrate=calibrate,
        track_progress_every_n=track_progress_every_n,
    )
    for multiscale_kwarg_name, multiscale_kwarg_value in multiscale_lddmm_kwargs.items():
        multiscale_lddmm_kwargs[multiscale_kwarg_name] = _validate_scalar_to_multi(multiscale_kwarg_value, size=len(multiscales), dtype=None)
    # Each value in the multiscale_lddmm_kwargs dictionary is an array with shape (len(multiscales)).

    for scale_index, scale in enumerate(multiscales):

        # Extract appropriate multiscale_lddmm_kwargs.
        this_scale_lddmm_kwargs = dict(map(lambda kwarg_name: (kwarg_name, multiscale_lddmm_kwargs[kwarg_name][scale_index]), multiscale_lddmm_kwargs.keys()))

        # rescale images and resolutions.
        # template.
        template_scale = np.round(scale * template.shape) / template.shape
        scaled_template = rescale(template, template_scale)
        scaled_template_resolution = template_resolution / template_scale
        # target.
        target_scale = np.round(scale * target.shape) / target.shape
        scaled_target = rescale(target, target_scale)
        scaled_target_resolution = target_resolution / target_scale

        # Collect non-multiscale_lddmm_kwargs
        multiscale_exempt_lddmm_kwargs = dict(
            # Images.
            template=scaled_template,
            target=scaled_target,
            # Image resolutions.
            template_resolution=scaled_template_resolution,
            target_resolution=scaled_target_resolution,

            # Initial values.
            initial_affine=initial_affine,
            initial_contrast_coefficients=initial_contrast_coefficients,
            initial_velocity_fields=initial_velocity_fields,
        )

        # Perform registration.

        # Set up _Lddmm instance.
        lddmm = _Lddmm(**this_scale_lddmm_kwargs, **multiscale_exempt_lddmm_kwargs)

        lddmm_dict = lddmm.register()

        # Overwrite initials for next scale if applicable.
        if scale_index < len(multiscales) - 1:
            # initial_affine.
            initial_affine = lddmm_dict['affine']
            # initial_contrast_coefficients.
            if multiscale_lddmm_kwargs['spatially_varying_contrast_map'][scale_index + 1] and multiscale_lddmm_kwargs['spatially_varying_contrast_map'][scale_index]:
                # If spatially_varying_contrast_map at next scale and at this scale, resize contrast_coefficients.
                next_target_shape = np.round(multiscales[scale_index + 1] * target.shape)
                initial_contrast_coefficients = resize(lddmm_dict['contrast_coefficients'], (*next_target_shape, multiscale_lddmm_kwargs['contrast_order'][scale_index + 1] + 1))
            elif not multiscale_lddmm_kwargs['spatially_varying_contrast_map'][scale_index + 1] and multiscale_lddmm_kwargs['spatially_varying_contrast_map'][scale_index]:
                    # If spatially_varying_contrast_map at this scale but not at next scale, average contrast_coefficients.
                    initial_contrast_coefficients = np.mean(lddmm_dict['contrast_coefficients'], axis=np.arange(template.ndim))
            else:
                # If spatially_varying_contrast_map at next scale but not this scale or at neither scale, initialize directly.
                initial_contrast_coefficients = lddmm_dict['contrast_coefficients']
            # initial_velocity_fields.
            next_template_shape = np.round(multiscales[scale_index + 1] * template.shape)
            initial_velocity_fields = sinc_resample(lddmm_dict['velocity_fields'], new_shape=(*next_template_shape, multiscale_lddmm_kwargs['num_timesteps'][scale_index + 1] or lddmm.num_timesteps, template.ndim))
        
        # End multiscales loop.
    
    return lddmm_dict


def generate_position_field(
    affine,
    velocity_fields,
    velocity_field_resolution,
    template_shape,
    template_resolution,
    target_shape,
    target_resolution,
    deform_to="template",
):
    """
    Integrate velocity_fields and apply affine to produce a position field.

    Parameters
    ----------
    affine : np.ndarray
        The affine array to be incorporated into the returned position field.
    velocity_fields : np.ndarray
        The velocity_fields defining the diffeomorphic flow. The leading dimensions are spatial, and the last two dimensions are the number of time steps and the coordinates.
    velocity_field_resolution : float, seq
        The resolution of velocity_fields, with multiple values given to specify anisotropy.
    template_shape : seq
        The shape of the template.
    template_resolution : float, seq
        The resolution of the template, with multiple values given to specify anisotropy.
    target_shape : seq
        The shape of the target.
    target_resolution : float, seq
        The resolution of the target, with multiple values given to specify anisotropy.
    deform_to : str, optional
        The direction of the deformation. By default "template".

    Returns
    -------
    np.ndarray
        The position field for the registration in the space of the image specified by deform_to.

    Raises
    ------
    ValueError
        Raised if the leading dimensions of velocity_fields fail to match template_shape.
    TypeError
        Raised if deform_to is not of type str.
    ValueError
        Raised if deform_to is neither 'template' nor 'target'.
    """

    # Validate inputs.
    # Validate template_shape. Not rigorous.
    template_shape = _validate_ndarray(template_shape)
    # Validate target_shape. Not rigorous.
    target_shape = _validate_ndarray(target_shape)
    # Validate velocity_fields.
    velocity_fields = _validate_ndarray(velocity_fields, required_ndim=len(template_shape) + 2)
    if not np.all(velocity_fields.shape[:-2] == template_shape):
        raise ValueError(f"velocity_fields' initial dimensions must equal template_shape.\n"
            f"velocity_fields.shape: {velocity_fields.shape}, template_shape: {template_shape}.")
    # Validate velocity_field_resolution.
    velocity_field_resolution = _validate_resolution(velocity_field_resolution, velocity_fields.ndim - 2)
    # Validate affine.
    affine = _validate_ndarray(affine, required_ndim=2, reshape_to_shape=(len(template_shape) + 1, len(template_shape) + 1))
    # Verify deform_to.
    if not isinstance(deform_to, str):
        raise TypeError(f"deform_to must be of type str.\n"
            f"type(deform_to): {type(deform_to)}.")
    elif deform_to not in ["template", "target"]:
        raise ValueError(f"deform_to must be either 'template' or 'target'.")

    # Compute intermediates.
    num_timesteps = velocity_fields.shape[-2]
    delta_t = 1 / num_timesteps
    template_axes = _compute_axes(template_shape, template_resolution)
    template_coords = _compute_coords(template_shape, template_resolution)
    target_axes = _compute_axes(target_shape, target_resolution)
    target_coords = _compute_coords(target_shape, target_resolution)

    # Create position field.
    if deform_to == "template":
        phi = np.copy(template_coords)
    elif deform_to == "target":
        phi_inv = np.copy(template_coords)

    # Integrate velocity field.
    for timestep in (reversed(range(num_timesteps)) if deform_to == "template" else range(num_timesteps)):
        if deform_to == "template":
            sample_coords = template_coords + velocity_fields[..., timestep, :] * delta_t
            phi = interpn(
                points=template_axes,
                values=phi - template_coords,
                xi=sample_coords,
                bounds_error=False,
                fill_value=None,
            ) + sample_coords
        elif deform_to == "target":
            sample_coords = template_coords - velocity_fields[..., timestep, :] * delta_t
            phi_inv = interpn(
                points=template_axes,
                values=phi_inv - template_coords,
                xi=sample_coords,
                bounds_error=False,
                fill_value=None,
            ) + sample_coords

    # Apply the affine transform to the position field.
    if deform_to == "template":
        # Apply the affine by multiplication.
        affine_phi = _multiply_coords_by_affine(affine, phi)
        # affine_phi has the resolution of the template.
    elif deform_to == "target":
        # Apply the affine by interpolation.
        sample_coords = _multiply_coords_by_affine(inv(affine), target_coords)
        phi_inv_affine_inv = interpn(
            points=template_axes,
            values=phi_inv - template_coords,
            xi=sample_coords,
            bounds_error=False,
            fill_value=None,
        ) + sample_coords
        # phi_inv_affine_inv has the resolution of the target.

    # return appropriate position field.
    if deform_to == "template":
        return affine_phi
    elif deform_to == "target":
        return phi_inv_affine_inv


def _transform_image(
    subject,
    subject_resolution,
    output_resolution,
    output_shape,
    position_field,
    position_field_resolution,
    extrapolation_fill_value=None,
):

    # Validate inputs.

    # Validate position_field.
    position_field = _validate_ndarray(position_field)
    # Validate position_field_resolution.
    position_field_resolution = _validate_resolution(position_field_resolution, position_field.ndim - 1)
    # Validate subject.
    subject = _validate_ndarray(subject, required_ndim=position_field.ndim - 1)
    # Validate subject_resolution.
    subject_resolution = _validate_resolution(subject_resolution, subject.ndim)
    # Validate output_resolution.
    if output_resolution is not None:
        output_resolution = _validate_resolution(output_resolution, subject.ndim)

    # Resample position_field if necessary.
    if output_resolution is not None and output_shape is not None:
        raise RuntimeError(f"Both output_resolution and output_shape were provided. Only one may be provided.")
    if output_resolution is not None:
        # resample position_field to match output_resolution.
        position_field = resample(
            image=position_field, 
            new_resolution=output_resolution, 
            old_resolution=position_field_resolution, 
            err_to_larger=True, 
            extrapolation_fill_value=None, 
            image_is_coords=True, 
        )
    else:
        # resize position_field to match output_shape.
        position_field = resize(position_field, output_shape)

    # Interpolate subject at position field.
    deformed_subject = interpn(
        points=_compute_axes(shape=subject.shape, resolution=subject_resolution),
        values=subject,
        xi=position_field,
        bounds_error=False,
        fill_value=extrapolation_fill_value,
    )

    return deformed_subject


def lddmm_transform_image(
    subject,
    subject_resolution=1,
    output_resolution=None,
    output_shape=None,
    deform_to="template",
    extrapolation_fill_value=None,
    # lddmm_register output (lddmm_dict).
    affine_phi=None,
    phi_inv_affine_inv=None,
    template_resolution=1,
    target_resolution=1,
    **unused_kwargs,
):
    """
    Apply the transform, or position field affine_phi or phi_inv_affine_inv, to the subject 
    to deform it to either the template or the target.

    The user is expected to provide subject, and optionally subject_resolution, deform_to, and output_resolution.
    It is expected that the rest of the arguments will be provided by keyword argument from the output of the register function.

    Example use:
        register_output_dict = lddmm_register(\*args, \*\*kwargs)
        deformed_subject = lddmm_transform_image(subject, subject_resolution, \*\*register_output_dict)

    Parameters
    ----------
        subject: np.ndarray
            The image to be deformed to the template or target from the results of the register function.
        subject_resolution: float, seq, optional
            The resolution of subject in each dimension, or just one scalar to indicate isotropy. By default 1.
        output_resolution: float, seq, optional
            The resolution of the output deformed_subject in each dimension, or just one scalar to indicate isotropy, 
            or None to indicate the resolution of template or target based on deform_to. Cannot be provided along with output_shape. By default None.
        output_shape: seq, optional
            The shape of the output deformed_subject, or None to indicate the shape of the template or target based on deform_to. Cannot be provided along with output_resolution. By default None.
        deform_to: str, optional
            Either "template" or "target", indicating which position field to apply to subject. By default "template".
        extrapolation_fill_value: float, optional
            The fill_value kwarg passed to scipy.interpolate.interpn. 
            If None, this is set to a low quantile of the subject's 10**-subject.ndim quantile to estimate background. By default None.
        affine_phi: np.ndarray, optional
            The position field in the shape of the template for deforming images to the template space. By default None.
        phi_inv_affine_inv: np.ndarray, optional
            The position field in the shape of the target for deforming images to the target space. By default None.
        template_resolution: float, seq, optional
            The resolution of the template in each dimension, or just one scalar to indicate isotropy. By default 1.
        target_resolution: float, seq, optional
            The resolution of the target in each dimension, or just one scalar to indicate isotropy. By default 1.

    Returns
    -------
    np.ndarray
        The result of applying the appropriate position field to subject, deforming it based on deform_to.

    Raises
    ------
    TypeError
        Raised if deform_to is not of type str.
    ValueError
        Raised if deform_to is a string other than "template" or "target".
    ValueError
        Raised if deform_to=="template" and affine_phi is None or deform_to=="target" and phi_inv_affine_inv is None.
    """

    # Validate inputs: subject, subject_resolution, deform_to, output_resolution, & extrapolation_fill_value.

    # Validate subject.
    subject = _validate_ndarray(subject)
    # Validate subject_resolution.
    subject_resolution = _validate_resolution(subject_resolution, subject.ndim)
    # Verify deform_to.
    if not isinstance(deform_to, str):
        raise TypeError(f"deform_to must be of type str.\n"
            f"type(deform_to): {type(deform_to)}.")
    elif deform_to not in ["template", "target"]:
        raise ValueError(f"deform_to must be either 'template' or 'target'.")
    # Validate output_resolution.
    if output_resolution is not None:
        output_resolution = _validate_resolution(output_resolution, subject.ndim)
    # Validate output_shape.
    if output_shape is not None:
        output_shape = _validate_ndarray(output_shape, required_shape=subject.ndim)
    # Validate extrapolation_fill_value.
    if extrapolation_fill_value is None:
        extrapolation_fill_value = np.quantile(subject, 10**-subject.ndim)

    # Define position_field and position_field_resolution.

    if deform_to == "template":
        position_field = affine_phi
        position_field_resolution = np.copy(template_resolution)
    elif deform_to == "target":
        position_field = phi_inv_affine_inv
        position_field_resolution = np.copy(target_resolution)
    # Verify position_field is not None.
    if position_field is None:
        raise ValueError(f"If deform_to=='template', affine_phi must be provided. If deform_to=='target', phi_inv_affine_inv must be provided.\n"
            f"deform_to: {deform_to}, affine_phi is None: {affine_phi is None}, phi_inv_affine_inv is None: {phi_inv_affine_inv is None}.")

    # Call _transform_image.

    deformed_subject = _transform_image(subject, subject_resolution, output_resolution, output_shape, position_field, position_field_resolution, extrapolation_fill_value)

    return deformed_subject


def _transform_points(
    points,
    position_field,
    position_field_resolution,
):

    # Validate inputs.

    # Validate position_field.
    position_field = _validate_ndarray(position_field)
    # Validate position_field_resolution.
    position_field_resolution = _validate_resolution(position_field_resolution, position_field.ndim - 1)
    # Validate points.
    points = _validate_ndarray(points, minimum_ndim=1)
    if points.shape[-1] != position_field.ndim - 1:
        raise ValueError(f"The length of the last dimension of points must match the spatial-dimensionality of position_field, i.e. position_field.ndim - 1.\n"
                         f"points.shape[-1]: {points.shape[-1]}, position_field.ndim - 1: {position_field.ndim - 1}.")

    # Interpolate points at position_field.
    transformed_points = interpn(
        points=_compute_axes(shape=position_field.shape[:-1], resolution=position_field_resolution),
        values=position_field,
        xi=points,
        bounds_error=False,
        fill_value=None,
    )

    return transformed_points


def lddmm_transform_points(
    points,
    deform_to="template",
    # lddmm_register output (lddmm_dict).
    affine_phi=None,
    phi_inv_affine_inv=None,
    template_resolution=1,
    target_resolution=1,
    **unused_kwargs,
):
    """
    Apply the transform, or position_field, to an array of points to transform them between the template and target spaces, as determined by deform_to.

    Parameters
    ----------
        points: np.ndarray
            The points in either the template space or the target space to be transformed into the other space, in physical units centered on the image. 
            The last dimension of points must have length equal to the dimensionality of the template and target.
        deform_to: str, optional
            Either "template" or "target" indicating whether to transform points to the template space or the target space. By default "template".
        affine_phi: np.ndarray, optional
            The position field in the shape of the template for deforming points to the target space. By default None.
        phi_inv_affine_inv: np.ndarray, optional
            The position field in the shape of the target for deforming points to the template space. By default None.
        template_resolution: float, seq, optional
            The resolution of the template in each dimension, or just one scalar to indicate isotropy. By default 1.
        target_resolution: float, seq, optional
            The resolution of the target in each dimension, or just one scalar to indicate isotropy. By default 1.

    Returns
    -------
    np.ndarray
        A copy of points transformed into the space determined by deform_to.

    Raises
    ------
    TypeError
        Raised if deform_to is not of type str.
    ValueError
        Raised if deform_to is neither "template" nor "target".
    """
    
    if not isinstance(deform_to, str):
    # Verify deform_to.
        raise TypeError(f"deform_to must be of type str.\n"
            f"type(deform_to): {type(deform_to)}.")
    elif deform_to not in ["template", "target"]:
        raise ValueError(f"deform_to must be either 'template' or 'target'.")

    # Define position_field and position_field_resolution.

    # Note: these are the reverse of what they are for lddmm_transform_image.
    if deform_to == "template":
        position_field = phi_inv_affine_inv
        position_field_resolution = np.copy(target_resolution)
    else:
        position_field = affine_phi
        position_field_resolution = np.copy(template_resolution)

    # Call _transform_points.

    transformed_points = _transform_points(points, position_field, position_field_resolution)

    return transformed_points
