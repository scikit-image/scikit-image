import warnings
import numpy as np
from scipy.interpolate import interpn
from scipy.linalg import inv, solve, det
from scipy.sparse.linalg import cg, LinearOperator
from matplotlib import pyplot as plt

from ._lddmm_utilities import _validate_ndarray
from ._lddmm_utilities import _validate_scalar_to_multi
from ._lddmm_utilities import _validate_resolution
from ._lddmm_utilities import _compute_axes
from ._lddmm_utilities import _compute_coords
from ._lddmm_utilities import _multiply_coords_by_affine
from ._lddmm_utilities import _compute_tail_determinant
from ._lddmm_utilities import resample

r'''
  _            _       _                         
 | |          | |     | |                        
 | |        __| |   __| |  _ __ ___    _ __ ___  
 | |       / _` |  / _` | | '_ ` _ \  | '_ ` _ \ 
 | |____  | (_| | | (_| | | | | | | | | | | | | |
 |______|  \__,_|  \__,_| |_| |_| |_| |_| |_| |_|
                                                 
'''

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
        # Stepsizes.
        affine_stepsize=None,
        deformative_stepsize=None,
        # Velocity field specifiers.
        sigma_regularization=None,
        smooth_length=None,
        num_timesteps=None,
        # Contrast map specifiers.
        contrast_order=None,
        spatially_varying_contrast_map=None,
        contrast_maxiter=None,
        contrast_tolerance=None,
        sigma_contrast=None,
        # Artifact specifiers.
        check_artifacts=None,
        sigma_artifact=None,
        # Smoothness vs. accuracy tradeoff.
        sigma_matching=None,
        # Initial values.
        initial_affine=None,
        initial_velocity_fields=None,
        initial_contrast_coefficients=None,
        # Diagnostic outputs.
        calibrate=None,
        track_progress_every_n=None,
    ):    

        # Constant inputs.

        # Images.
        self.template = _validate_ndarray(template)
        self.target = _validate_ndarray(target, required_ndim=self.template.ndim)

        # Resolution.
        self.template_resolution = _validate_scalar_to_multi(template_resolution if template_resolution is not None else 1, self.template.ndim)
        self.target_resolution = _validate_scalar_to_multi(target_resolution if target_resolution is not None else 1, self.target.ndim)

        # Iterations.
        self.num_iterations = int(num_iterations) if num_iterations is not None else 200
        self.num_affine_only_iterations = int(num_affine_only_iterations) if num_affine_only_iterations is not None else 50
        # Stepsizes.
        self.affine_stepsize = float(affine_stepsize) if affine_stepsize is not None else 0.2
        self.deformative_stepsize = float(deformative_stepsize) if deformative_stepsize is not None else 0
        # Velocity field specifiers.
        self.sigma_regularization = float(sigma_regularization) if sigma_regularization is not None else 10 * np.max(self.template_resolution)
        self.smooth_length = float(smooth_length) if smooth_length is not None else 2 * np.max(self.template_resolution)
        self.num_timesteps = int(num_timesteps) if num_timesteps is not None else 5
        # Contrast map specifiers.
        self.contrast_order = int(contrast_order) if contrast_order else 1
        if self.contrast_order < 1: raise ValueError(f"contrast_order must be at least 1.\ncontrast_order: {self.contrast_order}")
        self.spatially_varying_contrast_map = bool(spatially_varying_contrast_map) if spatially_varying_contrast_map is not None else False
        self.contrast_maxiter = int(contrast_maxiter) if contrast_maxiter else 5
        self.contrast_tolerance = float(contrast_tolerance) if contrast_tolerance else 1e-5
        self.sigma_contrast = float(sigma_contrast) if sigma_contrast else 1e-2
        # Artifact specifiers.
        self.check_artifacts = bool(check_artifacts) if check_artifacts is not None else False
        self.sigma_artifact = float(sigma_artifact) if sigma_artifact else 5 * float(sigma_matching) if sigma_matching else np.std(self.target) # Default: 5 * self.sigma_matching.
        # Smoothness vs. accuracy tradeoff.
        self.sigma_matching = float(sigma_matching) if sigma_matching else np.std(self.target)
        # Diagnostic outputs.
        self.calibrate = bool(calibrate) if calibrate is not None else False
        self.track_progress_every_n = int(track_progress_every_n) if track_progress_every_n is not None else 0

        # Constructions.

        # Constants.
        self.template_axes = _compute_axes(self.template.shape, self.template_resolution)
        self.template_coords = _compute_coords(self.template.shape, self.template_resolution)
        self.target_axes = _compute_axes(self.target.shape, self.target_resolution)
        self.target_coords = _compute_coords(self.target.shape, self.target_resolution)
        self.artifact_mean_value = np.max(self.target) if self.sigma_artifact is not None else 0 # TODO: verify this is right.
        self.fourier_high_pass_filter_power = 2
        fourier_velocity_fields_coords = _compute_coords(self.template.shape, 1 / (self.template_resolution * self.template.shape), origin='zero')
        self.fourier_high_pass_filter = (
            1 - self.smooth_length**2 
            * np.sum((-2  + 2 * np.cos(2 * np.pi * fourier_velocity_fields_coords * self.template_resolution)) / self.template_resolution**2, axis=-1)
        )**self.fourier_high_pass_filter_power
        self.delta_t = 1 / self.num_timesteps

        # Dynamics.
        if initial_affine is None:
            initial_affine = np.eye(template.ndim + 1)
        self.affine = _validate_ndarray(initial_affine, required_ndim=2, reshape_to_shape=(self.template.ndim + 1, self.template.ndim + 1))
        self.velocity_fields = initial_velocity_fields if initial_velocity_fields is not None else np.zeros((*self.template.shape, self.num_timesteps, self.template.ndim))
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
                self.contrast_coefficients = _validate_ndarray(initial_contrast_coefficients, reshape_to_shape=(*self.target.shape, self.contrast_order + 1))
        else:
            if initial_contrast_coefficients is None:
                self.contrast_coefficients = np.zeros(self.contrast_order + 1)
            else:
                self.contrast_coefficients = _validate_ndarray(initial_contrast_coefficients, reshape_to_shape=(self.contrast_order + 1))
        self.contrast_coefficients[..., 0] = np.mean(self.target) - np.mean(self.template) * np.std(self.target) / np.std(self.template)
        if self.contrast_order > 1: self.contrast_coefficients[..., 1] = np.std(self.target) / np.std(self.template)
        self.contrast_polynomial_basis = np.empty((*self.target.shape, self.contrast_order + 1))
        for power in range(self.contrast_order + 1):
            self.contrast_polynomial_basis[..., power] = self.deformed_template**power
        self.contrast_deformed_template = None
        fourier_template_coords = _compute_coords(self.template.shape, 1 / (self.template_resolution * self.template.shape), origin='zero')
        self.low_pass_filter = 1 / (
            (1 - self.smooth_length**2 * (
                np.sum((-2 + 2 * np.cos(2 * np.pi * self.template_resolution * fourier_template_coords)) / self.template_resolution**2, -1)
                )
            )**self.fourier_high_pass_filter_power
        )**2

        # Accumulators.
        self.matching_energies = []
        self.regularization_energies = []
        self.total_energies = []
        # For optional calibration plots.
        if self.calibrate:
            self.affines = []
            self.maximum_velocities = [0] * self.num_affine_only_iterations


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
                print(f"Progress: iteration {iteration}/{self.num_iterations}{' affine only' if iteration < self.num_affine_only_iterations else ''}.")

            # Forward pass: apply transforms to the template and compute the costs.

            # Compute position_field from velocity_fields.
            self._update_and_apply_position_field()
            # Contrast transform the deformed_template.
            self._apply_contrast_map()
            # Compute weights. 
            # This is the expectation step of the expectation maximization algorithm.
            if self.check_artifacts and iteration % 1 == 0: self._compute_weights()
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
            self._update_affine(affine_inv_gradient)
            # Update velocity_fields.
            if iteration >= self.num_affine_only_iterations: self._update_velocity_fields(velocity_fields_gradients)
        
        # Compute affine_phi in case there were only affine-only iterations.
        self._compute_affine_phi()

        # Optionally display useful plots for calibrating the registration parameters.
        if self.calibrate:
            self._generate_calibration_plots()
        
        # Note: the user-level apply_lddmm function relies on many of these specific outputs with these specific keys to function. 
        # ----> Check the apply_lddmm function signature before adjusting these outputs.
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
            regularization_energies=self.matching_energies,
            total_energies=self.total_energies,

            # Debuggers.
            lddmm=self,
        )


    def _update_and_apply_position_field(self):
        """
        Calculate phi_inv from v
        Compose on the right with Ainv
        Apply phi_invAinv to template

        Updates attributes:
            phi_inv
            deformed_template_to_time
            phi_inv_affine_inv
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

        Updates attributes:
            contrast_deformed_template
        """

        self.contrast_deformed_template = np.sum(self.contrast_polynomial_basis * self.contrast_coefficients, axis=-1)


    def _compute_weights(self):
        """
        Compute the matching_weights between the contrast_deformed_template and the target.

        Updates attributes:
            artifact_mean_value
            matching_weights
        """
        # TODO: rename.
        
        self.artifact_mean_value = np.mean(self.target * (1 - self.matching_weights)) / np.mean(1 - self.matching_weights)
        
        likelihood_matching = np.exp((self.contrast_deformed_template - self.target)**2 * (-1/(2 * self.sigma_matching**2))) / np.sqrt(2 * np.pi * self.sigma_matching**2)
        likelihood_artifact = np.exp((self.artifact_mean_value        - self.target)**2 * (-1/(2 * self.sigma_artifact**2))) / np.sqrt(2 * np.pi * self.sigma_artifact**2)

        self.matching_weights = likelihood_matching / (likelihood_matching + likelihood_artifact)


    def _compute_cost(self):
        """
        Compute the matching cost using a weighted sum of square error.

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
            np.sum(np.sum(np.abs(self.fourier_velocity_fields)**2, axis=(-1,-2)) * self.fourier_high_pass_filter) * 
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

        Updates attributes:
            contrast_polynomial_basis
            contrast_coefficients
        """

        # Update self.contrast_polynomial_basis.
        for power in range(self.contrast_order + 1):
            self.contrast_polynomial_basis[..., power] = self.deformed_template**power

        if self.spatially_varying_contrast_map:
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
                regularization_matrix /= self.sigma_contrast**2

                return (matching_matrix @ contrast_coefficients[..., None] - regularization_matrix[...,None]).ravel()
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
            self.contrast_coefficients = solve(basis_transpose_basis, basis_transpose_target, assume_a='pos')


    def _compute_affine_inv_gradient(self):
        """
        Compute and return the affine_inv gradient.
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

        # deformed_template_gradient_broadcast  has shape (x,y,z,3,4).
        # homogenous_target_coords              has shape (x,y,z,4).

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
        affine_inv_hessian_approx = matching_affine_inv_gradient.reshape(*matching_affine_inv_gradient.shape[:-2], -1, 1)
        affine_inv_hessian_approx = affine_inv_hessian_approx @ affine_inv_hessian_approx.reshape(*affine_inv_hessian_approx.shape[:-2], 1, -1)
        affine_inv_hessian_approx = np.sum(affine_inv_hessian_approx, tuple(range(self.target.ndim)))

        # Solve for affine_inv_gradient.
        affine_inv_gradient = solve(affine_inv_hessian_approx, affine_inv_gradient_reduction, assume_a='pos').reshape(matching_affine_inv_gradient.shape[-2:])
        # Append a row of zeros at the end of the 0th dimension.
        zeros = np.zeros((1, self.target.ndim + 1))
        affine_inv_gradient = np.concatenate((affine_inv_gradient, zeros), 0)

        return affine_inv_gradient


    def _update_affine(self, affine_inv_gradient):
        """
        Update self.affine_inv and self.affine based on affine_inv_gradient.

        if self.calibrate, appends the current self.affine to self.affines.

        Updates attributes:
            affine
            affines
        """
        
        affine_inv = inv(self.affine)

        affine_inv -= affine_inv_gradient * self.affine_stepsize

        self.affine = inv(affine_inv)

        # Save affine for calibration plotting.
        if self.calibrate:
            self.affines.append(self.affine)


    def _compute_velocity_fields_gradients(self):
        """
        Compute and return the gradients of the self.velocity_fields.

        Updates attributes:
            phi
            affine_phi
        """

        matching_error_prime = (self.contrast_deformed_template - self.target) * self.matching_weights / self.sigma_matching**2
        contrast_map_prime = np.zeros_like(self.target, float)
        for power in range(1, self.contrast_order + 1):
            contrast_map_prime += power * self.deformed_template**(power - 1) * self.contrast_coefficients[..., power]
        d_matching_d_deformed_template = matching_error_prime * contrast_map_prime

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
                points=self.target_axes,
                values=d_matching_d_deformed_template,
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

            # To convert from derivative to gradient we smooth by applying a low-pass filter in the frequency domain.
            matching_cost_at_t_gradient = np.fft.fftn(d_matching_d_velocity_at_t, axes=tuple(range(self.template.ndim))) * np.expand_dims(self.low_pass_filter, -1)
            # Add the gradient of the regularization term.
            # TODO: grab from compute_cost.
            matching_cost_at_t_gradient += np.fft.fftn(self.velocity_fields[...,timestep,:], axes=tuple(range(self.template.ndim))) / self.sigma_regularization**2
            # Invert fourier transform back to the spatial domain.
            d_matching_d_velocity_at_t = np.fft.ifftn(matching_cost_at_t_gradient, axes=tuple(range(self.template.ndim))).real

            d_matching_d_velocities.insert(0, d_matching_d_velocity_at_t)

        return d_matching_d_velocities

    
    def _update_velocity_fields(self, velocity_fields_gradients):
        """
        Update self.velocity_fields based on velocity_fields_gradient.

        if self.calibrate, calculates and appends the maximum velocity to self.maximum_velocities.

        Updates attributes:
            velocity_fields
            maximum_velocities
        """

        for timestep in range(self.num_timesteps):
            self.velocity_fields[...,timestep,:] -= velocity_fields_gradients[timestep] * self.deformative_stepsize

        # Save maximum velocity for calibration plotting.
        if self.calibrate:
            maximum_velocity = np.sqrt(np.sum(self.velocity_fields**2, axis=-1)).max()
            self.maximum_velocities.append(maximum_velocity)

    
    def _compute_affine_phi(self):
        """
        Compute and set self.affine_phi. Called once in case there were no deformative iterations to set it.

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

#TODO: fix docstring example re affine_stepsize
def lddmm_register(
    # Images.
    template,
    target,
    # Image resolutions.
    template_resolution=None,
    target_resolution=None,
    # Iterations.
    num_iterations=None,
    num_affine_only_iterations=None,
    # Stepsizes.
    affine_stepsize=None,
    deformative_stepsize=None,
    # Velocity field specifiers.
    sigma_regularization=None,
    smooth_length=None,
    num_timesteps=None,
    # Contrast map specifiers.
    contrast_order=None,
    spatially_varying_contrast_map=None,
    contrast_maxiter=None,
    contrast_tolerance=None,
    sigma_contrast=None,
    # Artifact specifiers.
    check_artifacts=None,
    sigma_artifact=None,
    # Smoothness vs. accuracy tradeoff.
    sigma_matching=None,
    # Initial values.
    initial_affine=None,
    initial_velocity_fields=None,
    initial_contrast_coefficients=None,
    # Diagnostic outputs.
    calibrate=None,
    track_progress_every_n=None,
):
    """
    Compute a registration between template and target, to be applied with apply_lddmm.
    
    Args:
        template (np.ndarray): The ideally clean template image being registered to the target.
        target (np.ndarray): The potentially messier target image being registered to.
        template_resolution (float, list, optional): A scalar or list of scalars indicating the resolution of the template. Overrides 0 input. Defaults to 1.
        target_resolution (float, optional): A scalar or list of scalars indicating the resolution of the target. Overrides 0 input. Defaults to 1.
        num_iterations (int, optional): The total number of iterations. Defaults to 200.
        num_affine_only_iterations (int, optional): The number of iterations at the start of the process without deformative adjustments. Defaults to 50.
        affine_stepsize (float, optional): The stepsize for affine adjustments. Should be between 0 and 1. Defaults to 0.2.
        deformative_stepsize (float, optional): The stepsize for deformative adjustments. Defaults to 0.
        sigma_regularization (float, optional): A scalar indicating the freedom to deform. Overrides 0 input. Defaults to 10 * np.max(self.template_resolution).
        smooth_length (float, optional): The length scale of smoothing. Overrides 0 input. Defaults to 2 * np.max(self.template_resolution).
        num_timesteps (int, optional): The number of composed sub-transformations in the diffeomorphism. Overrides 0 input. Defaults to 5.
        contrast_order (int, optional): The order of the polynomial fit between the contrasts of the template and target. Overrides 0 input. Defaults to 1.
        spatially_varying_contrast_map (bool, optional): If True, uses a polynomial per voxel to compute the contrast map rather than a single polynomial. Defaults to False.
        contrast_maxiter (int, optional): The maximum number of iterations to converge toward the optimal contrast_coefficients if spatially_varying_contrast_map == True. Overrides 0 input. Defaults to 100.
        contrast_tolerance (float, optional): The tolerance for convergence to the optimal contrast_coefficients if spatially_varying_contrast_map == True. Defaults to 1e-5.
        sigma_contrast (float, optional): The scale of variation in the contrast_coefficients if spatially_varying_contrast_map == True. Overrides 0 input. Defaults to 1e-2.
        check_artifacts (bool, optional): If True, artifacts are jointly classified with registration using sigma_artifact. Defaults to False.
        sigma_artifact (float, optional): The level of expected variation between artifact and non-artifact intensities. Overrides 0 input. Defaults to 5 * sigma_matching.
        sigma_matching (float, optional): An estimate of the spread of the noise in the target, 
            representing the tradeoff between the regularity and accuracy of the registration, where a smaller value should result in a less smooth, more accurate result. 
            Typically it should be set to an estimate of the standard deviation of the noise in the image, particularly with artifacts. Overrides 0 input. Defaults to the standard deviation of the target.
        initial_affine (np.ndarray, optional): The affine array that the registration will begin with. Defaults to np.eye(template.ndim + 1).
        initial_velocity_fields (np.ndarray, optional): The velocity fields that the registration will begin with. Defaults to all zeros.
        initial_contrast_coefficients (np.ndarray, optional): The contrast coefficients that the registration will begin with. 
            If None, the 0th order coefficient(s) are set to np.mean(self.target) - np.mean(self.template) * np.std(self.target) / np.std(self.template), 
            if self.contrast_order > 1, the 1st order coefficient(s) are set to np.std(self.target) / np.std(self.template), 
            and all others are set to zero. Defaults to None.
        calibrate (bool, optional): A boolean flag indicating whether to accumulate additional intermediate values and display informative plots for calibration purposes. Defaults to False.
        track_progress_every_n (int, optional): If positive, a progress update will be printed every track_progress_every_n iterations of registration. Defaults to 0.
    
    Example:
        >>> import numpy as np
        >>> from scipy.ndimage import rotate
        >>> from skimage.registration import lddmm_register, apply_lddmm
        >>> # 
        >>> # Define images. The template is registered to the target image but both transformations are returned.
        >>> # template is a binary elliptic cylinder with semi-radii 6 and 8 in dimensions 1 and 2. The overall shape is (2, 21, 29).
        >>> # target is a 30 degree rotation of template in the (1,2) plane.
        >>> # 
        >>> template = np.array([[[(col-12)**2 + (row-12)**2 <= 8**2 for col in range(25)] for row in range(25)]]*2, int)
        >>> target = rotate(template, 30, (1,2))
        >>> # 
        >>> # Register the template to the target, then deform the template and target to match the other.
        >>> # 
        >>> lddmm_dict = lddmm_register(template, target, translational_stepsize = 0.00001, linear_stepsize = 0.00001, deformative_stepsize = 0.5)
        >>> deformed_target   = apply_lddmm(target,   deform_to='template', **lddmm_dict)
        >>> deformed_template = apply_lddmm(template, deform_to='target',   **lddmm_dict)

    Returns:
        dict: A dictionary containing all important saved quantities computed during the registration.
    """



    # Set up Lddmm instance.
    lddmm = _Lddmm(
        # Images.
        template=template,
        target=target,
        # Image resolutions.
        template_resolution=template_resolution,
        target_resolution=target_resolution,
        # Iterations.
        num_iterations=num_iterations,
        num_affine_only_iterations=num_affine_only_iterations,
        # Stepsizes.
        affine_stepsize=affine_stepsize,
        deformative_stepsize=deformative_stepsize,
        # Velocity field specifiers.
        sigma_regularization=sigma_regularization,
        smooth_length=smooth_length,
        num_timesteps=num_timesteps,
        # Contrast map specifiers.
        contrast_order=contrast_order,
        spatially_varying_contrast_map=spatially_varying_contrast_map,
        contrast_maxiter=contrast_maxiter,
        contrast_tolerance=contrast_tolerance,
        sigma_contrast=sigma_contrast,
        # Artifact specifiers.
        check_artifacts=check_artifacts,
        sigma_artifact=sigma_artifact,
        # # vs. accuracy tradeoff.
        sigma_matching=sigma_matching,
        # Initial values.
        initial_affine=initial_affine,
        initial_velocity_fields=initial_velocity_fields,
        initial_contrast_coefficients=initial_contrast_coefficients,
        # Diagnostic outputs.
        calibrate=calibrate,
        track_progress_every_n=track_progress_every_n,
    )

    return lddmm.register()


def _generate_position_field(
    affine,
    velocity_fields,
    velocity_field_resolution,
    template_shape,
    template_resolution,
    target_shape,
    target_resolution,
    deform_to="template",
):

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
    velocity_field_resolution = _validate_resolution(velocity_fields.ndim - 2, velocity_field_resolution)
    # Validate affine.
    affine = _validate_ndarray(affine, required_ndim=2, reshape_to_shape=(len(template_shape), len(template_shape)))
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


def _apply_position_field(
    subject,
    subject_resolution,
    output_resolution,
    position_field,
    position_field_resolution,
    extrapolation_fill_value=None,
):

    # Validate inputs.

    # Validate position_field.
    position_field = _validate_ndarray(position_field)
    # Validate position_field_resolution.
    position_field_resolution = _validate_resolution(position_field.ndim - 1, position_field_resolution)
    # Validate subject.
    subject = _validate_ndarray(subject, required_ndim=position_field.ndim - 1)
    # Validate subject_resolution.
    subject_resolution = _validate_resolution(subject.ndim, subject_resolution)
    # Validate output_resolution.
    output_resolution = _validate_resolution(subject.ndim, output_resolution)

    # Resample position_field.
    position_field = resample(
        image=position_field, 
        new_resolution=output_resolution, 
        old_resolution=position_field_resolution, 
        err_to_larger=True, 
        extrapolation_fill_value=None, 
        image_is_coords=True, 
    )

    # Interpolate subject at position field.
    deformed_subject = interpn(
        points=_compute_axes(shape=subject.shape, resolution=subject_resolution),
        values=subject,
        xi=position_field,
        bounds_error=False,
        fill_value=extrapolation_fill_value,
    )

    return deformed_subject


def apply_lddmm(
    subject,
    subject_resolution=1,
    output_resolution=None,
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
        deformed_subject = apply_lddmm(subject, subject_resolution, \*\*register_output_dict)

    Args:
        subject (np.ndarray): The image to be deformed to the template or target from the results of the register function.
        subject_resolution (float, seq, optional): The resolution of subject in each dimension, or just one scalar to indicate isotropy. Defaults to 1.
        output_resolution (float, seq, optional): The resolution of the output deformed_subject in each dimension, 
            or just one scalar to indicate isotropy, or None to indicate the resolution of template or target based on deform_to. 
            Defaults to None.
        deform_to (str, optional): Either "template" or "target", indicating which position field to apply to subject. Defaults to "template".
        extrapolation_fill_value (float, optional): The fill_value kwarg passed to scipy.interpolate.interpn. 
            If None, this is set to a low quantile of the subject's 10**-subject.ndim quantile to estimate background. Defaults to None.
        affine_phi (np.ndarray, optional): The position field in the shape of the template for deforming to the template. Defaults to None.
        phi_inv_affine_inv (np.ndarray, optional): The position field in the shape of the target for deforming to the target. Defaults to None.
        template_resolution (float, seq, optional): The resolution of the template in each dimension, or just one scalar to indicate isotropy. Defaults to 1.
        target_resolution (float, seq, optional): The resolution of the target in each dimension, or just one scalar to indicate isotropy. Defaults to 1.

    Raises:
        TypeError: Raised if deform_to is not of type str.
        ValueError: Raised if deform_to is a string other than "template" or "target".
        ValueError: Raised if deform_to=="template" and affine_phi is None or deform_to=="target" and phi_inv_affine_inv is None.

    Returns:
        np.ndarray: The result of applying the appropriate position field to subject, deforming it based on deform_to.
    """

    # Validate inputs: subject, subject_resolution, deform_to, output_resolution, & extrapolation_fill_value.

    # Validate subject.
    subject = _validate_ndarray(subject)
    # Validate subject_resolution.
    subject_resolution = _validate_resolution(subject.ndim, subject_resolution)
    # Verify deform_to.
    if not isinstance(deform_to, str):
        raise TypeError(f"deform_to must be of type str.\n"
            f"type(deform_to): {type(deform_to)}.")
    elif deform_to not in ["template", "target"]:
        raise ValueError(f"deform_to must be either 'template' or 'target'.")
    # Validate output_resolution.
    if output_resolution is None and deform_to == "template" or output_resolution == "template":
        output_resolution = np.copy(template_resolution)
    elif output_resolution is None and deform_to == "target" or output_resolution == "target":
        output_resolution = np.copy(target_resolution)
    else:
        output_resolution = _validate_resolution(subject.ndim, output_resolution)
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

    # Call _apply_position_field.

    deformed_subject = _apply_position_field(subject, subject_resolution, output_resolution, position_field, position_field_resolution, extrapolation_fill_value)

    return deformed_subject
