"""
This is an implementation of the LDDMM algorithm with modifications, written
by Devin Crowley and based on "Diffeomorphic registration with intensity
transformation and missing data: Application to 3D digital pathology of
Alzheimer's disease."

This paper extends on an older LDDMM paper, "Computing large deformation
metric mappings via geodesic flows of diffeomorphisms."

This is the more recent paper:
Tward, Daniel, et al. "Diffeomorphic registration with intensity
transformation and missing data: Application to 3D digital pathology of
Alzheimer's disease." Frontiers in neuroscience 14 (2020).
https://doi.org/10.3389/fnins.2020.00052

This is the original LDDMM paper:
Beg, M. Faisal, et al. "Computing large deformation metric mappings via
geodesic flows of diffeomorphisms." International journal of computer vision
61.2 (2005): 139-157.
https://doi.org/10.1023/B:VISI.0000043755.93987.aa
"""

import warnings
import numpy as np
from collections import namedtuple

import scipy.ndimage as ndi
from scipy.linalg import inv, solve, det, svd
from skimage.transform import resize, rescale

from ._lddmm_utilities import _validate_scalar_to_multi
from ._lddmm_utilities import _validate_ndarray
from ._lddmm_utilities import _compute_axes
from ._lddmm_utilities import _compute_coords
from ._lddmm_utilities import _multiply_coords_by_affine
from ._lddmm_utilities import _compute_tail_determinant
from ._lddmm_utilities import sinc_resample
from skimage._shared.fft import fftmodule

r"""
  _            _       _
 | |          | |     | |
 | |        __| |   __| |  _ __ ___    _ __ ___
 | |       / _` |  / _` | | '_ ` _ \  | '_ ` _ \
 | |____  | (_| | | (_| | | | | | | | | | | | | |
 |______|  \__,_|  \__,_| |_| |_| |_| |_| |_| |_|

"""


class _Lddmm:
    """
    Class for storing shared values and objects used in learning a
    registration at a single scale.
    """

    def __init__(
        self,
        # Images.
        reference_image,
        moving_image,
        # Image spacings.
        reference_image_spacing=None,
        moving_image_spacing=None,
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
        contrast_iterations=None,
        sigma_contrast=None,
        contrast_smooth_length=None,
        # Smoothness vs. accuracy tradeoff.
        sigma_matching=None,
        # Classification specifiers.
        artifact_and_background_classification=None,
        sigma_artifact=None,
        sigma_background=None,
        artifact_prior=None,
        background_prior=None,
        # Initial values.
        initial_affine=None,
        initial_contrast_coefficients=None,
        initial_velocity_fields=None,
        # Diagnostic accumulators.
        affines=None,
        maximum_velocities=None,
        matching_energies=None,
        regularization_energies=None,
        total_energies=None,
    ):

        # Constant inputs.

        # Images.
        self.float_dtype = np.float32 if reference_image.dtype == np.float32 else np.float64
        self.reference_image = _validate_ndarray(reference_image, dtype=self.float_dtype)
        self.moving_image = _validate_ndarray(
            moving_image, dtype=self.float_dtype, required_ndim=self.reference_image.ndim
        )

        # spacing.
        self.reference_image_spacing = _validate_scalar_to_multi(
            reference_image_spacing
            if reference_image_spacing is not None
            else 1,
            self.reference_image.ndim,
            self.float_dtype,
        )
        self.moving_image_spacing = _validate_scalar_to_multi(
            moving_image_spacing if moving_image_spacing is not None else 1,
            self.moving_image.ndim,
            self.float_dtype,
        )

        # Iterations.
        self.num_iterations = (
            int(num_iterations) if num_iterations is not None else 300
        )
        self.num_affine_only_iterations = (
            int(num_affine_only_iterations)
            if num_affine_only_iterations is not None
            else 100
        )
        self.num_rigid_affine_iterations = (
            int(num_rigid_affine_iterations)
            if num_rigid_affine_iterations is not None
            else 50
        )

        # Stepsizes.
        self.affine_stepsize = (
            float(affine_stepsize) if affine_stepsize is not None else 0.3
        )
        self.deformative_stepsize = (
            float(deformative_stepsize)
            if deformative_stepsize is not None
            else 0
        )

        # Affine specifiers.
        self.fixed_affine_scale = (
            float(fixed_affine_scale)
            if fixed_affine_scale is not None
            else None
        )

        # Velocity field specifiers.
        self.sigma_regularization = (
            float(sigma_regularization)
            if sigma_regularization is not None
            else np.inf
        )
        self.velocity_smooth_length = (
            float(velocity_smooth_length)
            if velocity_smooth_length is not None
            else 2 * np.max(self.reference_image_spacing)
        )
        self.preconditioner_velocity_smooth_length = (
            float(preconditioner_velocity_smooth_length)
            if preconditioner_velocity_smooth_length is not None
            else 0
        )  # Default is inactive.
        self.maximum_velocity_fields_update = (
            float(maximum_velocity_fields_update)
            if maximum_velocity_fields_update is not None
            else np.max(
                self.reference_image.shape * self.reference_image_spacing
            )
        )  # Default is effectively inactive.
        self.num_timesteps = (
            int(num_timesteps) if num_timesteps is not None else 5
        )

        # Contrast map specifiers.
        self.contrast_order = (
            int(contrast_order) if contrast_order is not None else 1
        )

        if self.contrast_order < 1:
            raise ValueError(
                "contrast_order must be at least 1.\n"
                f"contrast_order: {self.contrast_order}"
            )
        self.spatially_varying_contrast_map = (
            bool(spatially_varying_contrast_map)
            if spatially_varying_contrast_map is not None
            else False
        )
        self.contrast_iterations = (
            int(contrast_iterations) if contrast_iterations else 5
        )
        self.sigma_contrast = float(sigma_contrast) if sigma_contrast else 1
        self.contrast_smooth_length = (
            float(contrast_smooth_length)
            if contrast_smooth_length
            else 10 * np.max(self.moving_image_spacing)
        )

        # Smoothness vs. accuracy tradeoff.
        self.sigma_matching = (
            float(sigma_matching)
            if sigma_matching
            else np.std(self.moving_image)
        )

        # Classification specifiers.
        self.artifact_and_background_classification = (
            bool(artifact_and_background_classification)
            if artifact_and_background_classification is not None
            else False
        )
        self.sigma_artifact = (
            float(sigma_artifact)
            if sigma_artifact
            else 5 * self.sigma_matching
        )
        self.sigma_background = (
            float(sigma_background)
            if sigma_background
            else 2 * self.sigma_matching
        )
        self.artifact_prior = (
            float(artifact_prior) if artifact_prior is not None else 1 / 3
        )
        self.background_prior = (
            float(background_prior) if background_prior is not None else 1 / 3
        )
        if self.artifact_prior + self.background_prior >= 1:
            raise ValueError(
                "artifact_prior and background_prior must sum to less than 1."
            )

        # Diagnostic accumulators.
        self.affines = list(affines) if affines is not None else []
        self.maximum_velocities = (
            list(maximum_velocities) if maximum_velocities is not None else []
        )
        self.maximum_velocities += [0] * self.num_affine_only_iterations
        self.matching_energies = (
            list(matching_energies) if matching_energies is not None else []
        )
        self.regularization_energies = (
            list(regularization_energies)
            if regularization_energies is not None
            else []
        )
        self.total_energies = (
            list(total_energies) if total_energies is not None else []
        )

        # Constructions.

        # Constants.
        self.reference_image_axes = _compute_axes(
            self.reference_image.shape, self.reference_image_spacing,
            dtype=self.float_dtype
        )
        self.reference_image_coords = _compute_coords(
            self.reference_image.shape, self.reference_image_spacing,
            dtype=self.float_dtype
        )
        self.moving_image_axes = _compute_axes(
            self.moving_image.shape, self.moving_image_spacing,
            dtype=self.float_dtype
        )
        self.moving_image_coords = _compute_coords(
            self.moving_image.shape, self.moving_image_spacing,
            dtype=self.float_dtype
        )

        self.artifact_mean_value = np.max(self.moving_image)
        self.background_mean_value = np.min(self.moving_image)
        self.delta_t = 1 / self.num_timesteps
        self.fourier_filter_power = 2
        fourier_velocity_fields_coords = _compute_coords(
            self.reference_image.shape,
            1 / (self.reference_image_spacing * self.reference_image.shape),
            origin="zero",
            dtype=self.float_dtype,
        )
        self.fourier_high_pass_filter = (
            1
            - self.velocity_smooth_length ** 2
            * np.sum(
                (
                    -2
                    + 2
                    * np.cos(
                        2
                        * np.pi
                        * fourier_velocity_fields_coords
                        * self.reference_image_spacing
                    )
                )
                / self.reference_image_spacing ** 2,
                axis=-1,
            )
        ) ** self.fourier_filter_power
        fourier_reference_image_coords = _compute_coords(
            self.reference_image.shape,
            1 / (self.reference_image_spacing * self.reference_image.shape),
            origin="zero",
            dtype=self.float_dtype,
        )
        self.low_pass_filter = 1 / (
            (
                1
                - self.velocity_smooth_length ** 2
                * (
                    np.sum(
                        (
                            -2
                            + 2
                            * np.cos(
                                2
                                * np.pi
                                * self.reference_image_spacing
                                * fourier_reference_image_coords
                            )
                        )
                        / self.reference_image_spacing ** 2,
                        -1,
                    )
                )
            )
            ** (2 * self.fourier_filter_power)
        )
        # This filter affects the optimization but not the optimum.
        self.preconditioner_low_pass_filter = 1 / (
            (
                1
                - self.preconditioner_velocity_smooth_length ** 2
                * (
                    np.sum(
                        (
                            -2
                            + 2
                            * np.cos(
                                2
                                * np.pi
                                * self.reference_image_spacing
                                * fourier_reference_image_coords
                            )
                        )
                        / self.reference_image_spacing ** 2,
                        -1,
                    )
                )
            )
            ** (2 * self.fourier_filter_power)
        )
        fourier_moving_image_coords = _compute_coords(
            self.moving_image.shape,
            1 / (self.moving_image_spacing * self.moving_image.shape),
            origin="zero",
            dtype=self.float_dtype,
        )
        self.contrast_high_pass_filter = (
            1
            - self.contrast_smooth_length ** 2
            * (
                np.sum(
                    (
                        -2
                        + 2
                        * np.cos(
                            2
                            * np.pi
                            * self.moving_image_spacing
                            * fourier_moving_image_coords
                        )
                    )
                    / self.moving_image_spacing ** 2,
                    -1,
                )
            )
        ) ** self.fourier_filter_power / self.sigma_contrast

        # Dynamics.
        if initial_affine is None:
            initial_affine = np.eye(reference_image.ndim + 1,
                                    dtype=self.float_dtype)
        self.affine = _validate_ndarray(
            initial_affine,
            required_shape=(
                self.reference_image.ndim + 1,
                self.reference_image.ndim + 1,
            ),
        ).astype(self.float_dtype, copy=False)
        if initial_velocity_fields is not None:
            self.velocity_fields = _validate_ndarray(
                initial_velocity_fields,
                required_shape=(
                    *self.reference_image.shape,
                    self.num_timesteps,
                    self.reference_image.ndim,
                ),
            ).astype(self.float_dtype, copy=False)
        else:
            self.velocity_fields = np.zeros(
                (
                    *self.reference_image.shape,
                    self.num_timesteps,
                    self.reference_image.ndim,
                ),
                dtype=self.float_dtype,
            )
        # Note: If a transformation T maps a point in the space of the
        # reference_image to a point in the space of the moving_image, as
        # affine_phi, the reference_image image is deformed using T_inv, or
        # phi_inv_affine_inv via interpolation of the moving_image at
        # phi_inv_affine_inv.
        # phi: A position-field that describes change in shape, or
        # deformation, of the reference_image but not change in scale or
        # orientation. Stored in reference_image-space.
        self.phi = np.copy(self.reference_image_coords)
        # affine_phi: A position-field that composes of phi then affine that
        # describes change in shape, i.e. deformation, and change in scale
        # and orientation. Stored in reference_image-space. This is used for
        # transforming images in moving_image-space to reference_image-space by
        # interpolation.
        self.affine_phi = np.copy(self.reference_image_coords)
        # phi_inv: A position-field that describes the inverse change in
        # shape, or deformation, of the reference_image but not change in scale
        # or orientation. Stored in reference_image-space.
        self.phi_inv = np.copy(self.reference_image_coords)
        # phi_inv_affine_inv: A position-field that composes affine_inv then
        # phi_inv that describes change in shape, i.e. deformation, and
        # change in scale and orientation. Stored in moving_image-space. This
        # is used for transforming images in reference_image-space to
        # moving_image-space by interpolation.
        self.phi_inv_affine_inv = np.copy(self.moving_image_coords)
        self.complex_float_dtype = np.promote_types(self.float_dtype,
                                                    np.complex64)
        self.fourier_velocity_fields = np.zeros_like(
            self.velocity_fields, self.complex_float_dtype
        )
        self.matching_weights = np.ones_like(self.moving_image)
        self.deformed_reference_image_to_time = []
        self.deformed_reference_image = ndi.map_coordinates(
            self.reference_image,
            self._normalize_coords(self.phi_inv_affine_inv),
            order=1,
            mode='nearest',
            output=self.float_dtype,
        )

        if spatially_varying_contrast_map:
            if initial_contrast_coefficients is None:
                self.contrast_coefficients = np.zeros(
                    (*self.moving_image.shape, self.contrast_order + 1),
                    dtype=self.float_dtype,
                )
            else:
                self.contrast_coefficients = _validate_ndarray(
                    initial_contrast_coefficients,
                    broadcast_to_shape=(
                        *self.moving_image.shape,
                        self.contrast_order + 1,
                    ),
                ).astype(self.float_dtype, copy=False)
        else:
            if initial_contrast_coefficients is None:
                self.contrast_coefficients = np.zeros(self.contrast_order + 1,
                                                      dtype=self.float_dtype)
            else:
                self.contrast_coefficients = _validate_ndarray(
                    initial_contrast_coefficients,
                    required_shape=(self.contrast_order + 1),
                ).astype(self.float_dtype, copy=False)
        self.contrast_coefficients[..., 0] = np.mean(
            self.moving_image
        ) - np.mean(self.reference_image) * np.std(self.moving_image) / np.std(
            self.reference_image
        )
        if self.contrast_order > 0:
            self.contrast_coefficients[..., 1] = np.std(
                self.moving_image
            ) / np.std(self.reference_image)
        self.contrast_polynomial_basis = np.empty(
            (*self.moving_image.shape, self.contrast_order + 1),
            dtype=self.float_dtype,
        )
        for power in range(self.contrast_order + 1):
            self.contrast_polynomial_basis[..., power] = (
                self.deformed_reference_image ** power
            )
        self.contrast_deformed_reference_image = np.sum(
            self.contrast_polynomial_basis * self.contrast_coefficients,
            axis=-1,
        )  # Initialized value not used.

        # Preempt known error.
        if np.any(np.array(self.reference_image.shape) == 1) or np.any(
            np.array(self.moving_image.shape) == 1
        ):
            raise RuntimeError(
                "Known issue: Images with a 1 in their shape are not "
                "supported by scipy.interpolate.interpn.\n"
                f"self.reference_image.shape: {self.reference_image.shape},"
                f"self.moving_image.shape: {self.moving_image.shape}.\n"
            )

    def register(self):
        """
        Register the reference_image to the moving_image using the current
        state of the attributes.

        Return a dictionary of relevant quantities most notably including the
        transformations:
            phi_inv_affine_inv is the position field that maps the
            reference_image to the moving_image.
            affine_phi is the position field that maps the moving_image to the
                reference_image.
        """

        # Iteratively perform each step of the registration.
        for iteration in range(self.num_iterations):

            # Forward pass: apply transforms to the reference_image and
            # compute the costs.

            # Compute position_field from velocity_fields.
            self._update_and_apply_position_field()
            # Contrast transform the deformed_reference_image.
            self._apply_contrast_map()
            # Compute weights.
            # This is the expectation step of the expectation maximization
            # algorithm.
            # Note: recomputing weights may be appropriate less frequently
            # than every iteration.
            if self.artifact_and_background_classification:
                self._compute_weights()
            # Compute cost.
            self._compute_cost()

            # Backward pass: update contrast map, affine, & velocity_fields.

            # Compute contrast map.
            self._compute_contrast_map()

            # Compute affine gradient.
            affine_inv_gradient = self._compute_affine_inv_gradient()
            # Compute velocity_fields gradient.
            if iteration >= self.num_affine_only_iterations:
                velocity_fields_gradients = (
                    self._compute_velocity_fields_gradients()
                )
            # Update affine.
            self._update_affine(
                affine_inv_gradient, iteration
            )  # rigid_only=iteration < self.rigid_only_iterations)
            # Update velocity_fields.
            if iteration >= self.num_affine_only_iterations:
                self._update_velocity_fields(velocity_fields_gradients)
        # End for loop.

        # Compute affine_phi in case there were only affine-only iterations.
        self._compute_affine_phi()

    def _update_and_apply_position_field(self):
        """
        Calculate phi_inv from v
        Compose on the right with Ainv
        Apply phi_inv_affine_inv to reference_image

        Accesses attributes:
            reference_image
            reference_image_axes
            reference_image_coords
            moving_image_coords
            num_timesteps
            delta_t
            affine
            velocity_fields
            phi_inv
            phi_inv_affine_inv
            deformed_reference_image_to_time
            deformed_reference_image

        Updates attributes:
            phi_inv
            phi_inv_affine_inv
            deformed_reference_image_to_time
            deformed_reference_image
        """

        # Set self.phi_inv to identity.
        self.phi_inv = np.copy(self.reference_image_coords)

        # Reset self.deformed_reference_image_to_time.
        self.deformed_reference_image_to_time = []
        for timestep in range(self.num_timesteps):
            # Compute phi_inv.
            sample_coords = (
                self.reference_image_coords
                - self.velocity_fields[..., timestep, :] * self.delta_t
            )
            for i in range(self.reference_image.ndim):
                self.phi_inv[..., i] = ndi.map_coordinates(
                    self.phi_inv[..., i] - self.reference_image_coords[..., i],
                    self._normalize_coords(sample_coords),
                    order=1,
                    mode='nearest',
                    output=self.float_dtype,
                ) + sample_coords[..., i]

            self.deformed_reference_image_to_time.append(
                ndi.map_coordinates(
                    self.reference_image,
                    self._normalize_coords(self.phi_inv),
                    order=1,
                    mode='nearest',
                    output=self.float_dtype,
                )
            )

            # End time loop.

        # Apply affine_inv to moving_image_coords by multiplication.
        affine_inv_moving_image_coords = _multiply_coords_by_affine(
            inv(self.affine), self.moving_image_coords
        )

        coords = self._normalize_coords(affine_inv_moving_image_coords)
        for i in range(self.reference_image.ndim):
            self.phi_inv_affine_inv[..., i] = ndi.map_coordinates(
                self.phi_inv[..., i] - self.reference_image_coords[..., i],
                coords,
                order=1,
                mode='nearest',
                output=self.float_dtype,
            ) + affine_inv_moving_image_coords[..., i]

        self.deformed_reference_image = ndi.map_coordinates(
            self.reference_image,
            self._normalize_coords(self.phi_inv_affine_inv),
            order=1,
            mode='nearest',
            output=self.float_dtype,
        )

    def _normalize_coords(self, coords):
        """
        Normalize coordinates relative to spacing of 1 for the reference image.

        This normalization allows use of scipy.ndimage.map_coordinates where
        the input image coordinates are assumed to correspond to integer
        coordinate locations.
        """
        return np.stack(
            [((coords[..., i] - self.reference_image_axes[i][0])
              / self.reference_image_spacing[i])
             for i in range(self.reference_image.ndim)
            ],
            axis=0
        )

    def _apply_contrast_map(self):
        """
        Apply contrast_coefficients to deformed_reference_image to produce
        contrast_deformed_reference_image.

        Accsses attributes:
            contrast_polynomial_basis
            contrast_coefficients
            contrast_deformed_reference_image


        Updates attributes:
            contrast_deformed_reference_image
        """

        self.contrast_deformed_reference_image = np.sum(
            self.contrast_polynomial_basis * self.contrast_coefficients,
            axis=-1,
        )

    def _compute_weights(self):
        """
        Compute the matching_weights between the
        contrast_deformed_reference_image and the moving_image.

        Accsses attributes:
            moving_image
            sigma_matching
            sigma_artifact
            sigma_background
            artifact_prior
            background_prior
            contrast_deformed_reference_image
            artifact_mean_value
            background_mean_value
            matching_weights

        Updates attributes:
            artifact_mean_value
            background_mean_value
            matching_weights
        """

        likelihood_matching = np.exp(
            (self.contrast_deformed_reference_image - self.moving_image) ** 2
            * (-1 / (2 * self.sigma_matching ** 2))
        ) / np.sqrt(2 * np.pi * self.sigma_matching ** 2)
        likelihood_artifact = np.exp(
            (self.artifact_mean_value - self.moving_image) ** 2
            * (-1 / (2 * self.sigma_artifact ** 2))
        ) / np.sqrt(2 * np.pi * self.sigma_artifact ** 2)
        likelihood_background = np.exp(
            (self.background_mean_value - self.moving_image) ** 2
            * (-1 / (2 * self.sigma_background ** 2))
        ) / np.sqrt(2 * np.pi * self.sigma_background ** 2)

        # Account for priors.
        likelihood_matching *= 1 - self.artifact_prior - self.background_prior
        likelihood_artifact *= self.artifact_prior
        likelihood_background *= self.background_prior

        # Where the denominator is less than 1e-6 of its maximum, set it to
        # 1e-6 of its maximum to avoid division by zero.
        likelihood_sum = (
            likelihood_matching + likelihood_artifact + likelihood_background
        )
        likelihood_sum_max = np.max(likelihood_sum)
        likelihood_sum[likelihood_sum < 1e-6 * likelihood_sum_max] = (
            1e-6 * likelihood_sum_max
        )

        self.matching_weights = likelihood_matching / likelihood_sum
        artifact_weights = likelihood_artifact / likelihood_sum
        background_weights = likelihood_background / likelihood_sum

        self.artifact_mean_value = np.mean(
            self.moving_image * artifact_weights
        ) / np.mean(artifact_weights)
        self.background_mean_value = np.mean(
            self.moving_image * background_weights
        ) / np.mean(background_weights)

    def _compute_cost(self):
        """
        Compute the matching cost using a weighted sum of square error.

        Accsses attributes:
            moving_image
            reference_image
            reference_image_spacing
            moving_image_spacing
            contrast_deformed_reference_image
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
            np.sum(
                (self.contrast_deformed_reference_image - self.moving_image)
                ** 2
                * self.matching_weights
            )
            * 1
            / (2 * self.sigma_matching ** 2)
            * np.prod(self.moving_image_spacing)
        )

        regularization_energy = np.sum(
            np.sum(np.abs(self.fourier_velocity_fields) ** 2, axis=(-1, -2))
            * self.fourier_high_pass_filter ** 2
        ) * (
            np.prod(self.reference_image_spacing)
            * self.delta_t
            / (2 * self.sigma_regularization ** 2)
            / self.reference_image.size
        )

        total_energy = matching_energy + regularization_energy

        # Accumulate energies.
        self.matching_energies.append(matching_energy)
        self.regularization_energies.append(regularization_energy)
        self.total_energies.append(total_energy)

    def _compute_contrast_map(self):
        """
        Compute contrast_coefficients mapping deformed_reference_image to
        moving_image.

        Accesses attributes:
            moving_image
            moving_image_spacing
            deformed_reference_image
            spatially_varying_contrast_map
            sigma_matching
            contrast_order
            sigma_contrast
            contrast_iterations
            matching_weights
            contrast_polynomial_basis
            contrast_coefficients

        Updates attributes:
            contrast_polynomial_basis
            contrast_coefficients
        """

        # Update self.contrast_polynomial_basis.
        for power in range(self.contrast_order + 1):
            self.contrast_polynomial_basis[..., power] = (
                self.deformed_reference_image ** power
            )

        if self.spatially_varying_contrast_map:
            # Compute and set self.contrast_coefficients for
            # self.spatially_varying_contrast_map == True.

            # C is contrast_coefficients.
            # B is the contrast_polynomial_basis.
            # W is weights.
            # L is contrast_high_pass_filter.
            # This is the minimization problem:
            # sum(|BC - J|^2 W^2 / 2) + sum(|LC|^2 / 2).
            # The linear equation we need to solve for C is this:
            # W^2 B^T B C  + L^T L C = W^2 B^T J.
            # Where W acts by pointwise multiplication, B acts by matrix
            # multiplication at every point, and L acts by filtering in the
            # Fourier domain.
            # Let L C = D. --> C = L^{-1} D.
            # This reformulates the problem to:
            # W^2 B^T B L^{-1} D + L^T D = W^2 B^T J.
            # Then, to make it nicer we act on both sides with L^{-T},
            # yielding:
            # L^{-T}(B^T B) L^{-1}D + D = L^{-T} W^2 B^t J.
            # Then we factor the left side:
            # [L^{-T} B^T  B L^{-1} + identity]D = L^{-T}W^2 B^T J

            spatial_ndim = self.contrast_polynomial_basis.ndim - 1

            # Represents W in the equation.
            weights = np.sqrt(self.matching_weights) / self.sigma_matching

            # Represents the right hand side of the equation.
            right_hand_side = (
                self.contrast_polynomial_basis
                * (weights ** 2 * self.moving_image)[..., None]
            )

            # Reformulate with block elimination.
            high_pass_contrast_coefficients = fftmodule.ifftn(
                fftmodule.fftn(
                    self.contrast_coefficients, axes=range(spatial_ndim)
                )
                * self.contrast_high_pass_filter[..., None],
                axes=range(spatial_ndim),
            ).real
            low_pass_right_hand_side = fftmodule.ifftn(
                fftmodule.fftn(right_hand_side, axes=range(spatial_ndim))
                / self.contrast_high_pass_filter[..., None],
                axes=range(spatial_ndim),
            ).real
            for _ in range(self.contrast_iterations):
                linear_operator_high_pass_contrast_coefficients = (
                    fftmodule.ifftn(
                        fftmodule.fftn(
                            (
                                np.sum(
                                    fftmodule.ifftn(
                                        fftmodule.fftn(
                                            high_pass_contrast_coefficients,
                                            axes=range(spatial_ndim),
                                        )
                                        / self.contrast_high_pass_filter[
                                            ..., None
                                        ],
                                        axes=range(spatial_ndim),
                                    ).real
                                    * self.contrast_polynomial_basis,
                                    axis=-1,
                                )
                                * weights ** 2
                            )[..., None]
                            * self.contrast_polynomial_basis,
                            axes=range(spatial_ndim),
                        )
                        / self.contrast_high_pass_filter[..., None],
                        axes=range(spatial_ndim),
                    ).real
                    + high_pass_contrast_coefficients
                )
                residual = (
                    linear_operator_high_pass_contrast_coefficients
                    - low_pass_right_hand_side
                )
                # Compute the optimal step size.
                linear_operator_residual = (
                    fftmodule.ifftn(
                        fftmodule.fftn(
                            (
                                np.sum(
                                    fftmodule.ifftn(
                                        fftmodule.fftn(
                                            residual, axes=range(spatial_ndim)
                                        )
                                        / self.contrast_high_pass_filter[
                                            ..., None
                                        ],
                                        axes=range(spatial_ndim),
                                    ).real
                                    * self.contrast_polynomial_basis,
                                    axis=-1,
                                )
                                * weights ** 2
                            )[..., None]
                            * self.contrast_polynomial_basis,
                            axes=range(spatial_ndim),
                        )
                        / self.contrast_high_pass_filter[..., None],
                        axes=range(spatial_ndim),
                    ).real
                    + residual
                )
                optimal_stepsize = np.sum(residual ** 2) / np.sum(
                    linear_operator_residual * residual
                )
                # Take gradient descent step at half the optimal step size.
                high_pass_contrast_coefficients -= (
                    optimal_stepsize * residual / 2
                )

            self.contrast_coefficients = fftmodule.ifftn(
                fftmodule.fftn(
                    high_pass_contrast_coefficients, axes=range(spatial_ndim)
                )
                / self.contrast_high_pass_filter[..., None],
                axes=range(spatial_ndim),
            ).real

        else:
            # Compute and set self.contrast_coefficients for
            # self.spatially_varying_contrast_map == False.

            # Ravel necessary components for convenient matrix multiplication.
            moving_image_ravel = np.ravel(self.moving_image)
            matching_weights_ravel = np.ravel(self.matching_weights)
            contrast_polynomial_basis_semi_ravel = np.reshape(
                self.contrast_polynomial_basis,
                (self.moving_image.size, -1),
            )  # A view, not a copy.

            # Create intermediate composites.
            basis_transpose_basis = np.matmul(
                contrast_polynomial_basis_semi_ravel.T
                * matching_weights_ravel,
                contrast_polynomial_basis_semi_ravel,
            )
            basis_transpose_moving_image = np.matmul(
                contrast_polynomial_basis_semi_ravel.T
                * matching_weights_ravel,
                moving_image_ravel,
            )

            # Solve for contrast_coefficients.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", message="Ill-conditioned matrix"
                )
                try:
                    self.contrast_coefficients = solve(
                        basis_transpose_basis,
                        basis_transpose_moving_image,
                        assume_a="pos",
                    )
                except np.linalg.LinAlgError as e:
                    raise np.linalg.LinAlgError(
                        "This exception may have been raised because the"
                        "contrast_polynomial_basis vectors were not"
                        "independent, i.e. the reference_image is constant."
                    ) from e

    def _compute_affine_inv_gradient(self):
        """
        Compute and return the affine_inv gradient.

        Accesss attributes:
            reference_image
            moving_image
            reference_image_spacing
            reference_image_axes
            moving_image_coords
            deformed_reference_image
            contrast_deformed_reference_image
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

        # Generate the reference_image image deformed by phi_inv but not
        # affected by the affine.
        non_affine_deformed_reference_image = ndi.map_coordinates(
            self.reference_image,
            self._normalize_coords(self.phi_inv),
            order=1,
            mode='nearest',
            output=self.float_dtype,
        )

        # Compute the gradient of non_affine_deformed_reference_image.
        non_affine_deformed_reference_image_gradient = np.stack(
            np.gradient(
                non_affine_deformed_reference_image,
                *self.reference_image_spacing,
            ),
            -1,
        )

        # Apply the affine to each component of
        # non_affine_deformed_reference_image gradient.
        sample_coords = _multiply_coords_by_affine(
            inv(self.affine), self.moving_image_coords
        )
        coords = self._normalize_coords(sample_coords)
        deformed_reference_image_gradient = np.empty(
            sample_coords.shape,
            dtype=non_affine_deformed_reference_image_gradient.dtype
        )
        for i in range(self.reference_image.ndim):
            deformed_reference_image_gradient[..., i] = ndi.map_coordinates(
                non_affine_deformed_reference_image_gradient[..., i],
                coords,
                order=1,
                mode='nearest',
                output=self.float_dtype,
            )

        # Reshape and broadcast deformed_reference_image_gradient from shape
        # (x,y,z,3) to (x,y,z,3,1) to (x,y,z,3,4) - for a 3D example.
        deformed_reference_image_gradient_broadcast = np.repeat(
            np.expand_dims(deformed_reference_image_gradient, -1),
            repeats=self.moving_image.ndim + 1,
            axis=-1,
        )

        # Construct homogenous_moving_image_coords by appending 1's at the end
        # of the last dimension throughout self.moving_image_coords.
        ones = np.ones((*self.moving_image.shape, 1))
        homogenous_moving_image_coords = np.concatenate(
            (self.moving_image_coords, ones), -1
        )

        # For a 3D example:

        # deformed_reference_image_gradient_broadcast has shape (x,y,z,3,4).
        # homogenous_moving_image_coords has shape (x,y,z,4).

        # To repeat homogenous_moving_image_coords along the 2nd-last
        # dimension of deformed_reference_image_gradient_broadcast, we reshape
        # homogenous_moving_image_coords from shape (x,y,z,4)
        # to shape (x,y,z,1,4) and let that broadcast to shape (x,y,z,3,4).

        matching_affine_inv_gradient = (
            deformed_reference_image_gradient_broadcast
            * np.expand_dims(homogenous_moving_image_coords, -2)
        )

        # Get error term.
        matching_error_prime = (
            (self.contrast_deformed_reference_image - self.moving_image)
            * self.matching_weights
            / self.sigma_matching ** 2
        )
        contrast_map_prime = np.zeros_like(self.moving_image, self.float_dtype)
        for power in range(1, self.contrast_order + 1):
            contrast_map_prime += (
                power
                * self.deformed_reference_image ** (power - 1)
                * self.contrast_coefficients[..., power]
            )
        d_matching_d_deformed_reference_image = (
            matching_error_prime * contrast_map_prime
        )

        affine_inv_gradient = (
            matching_affine_inv_gradient
            * d_matching_d_deformed_reference_image[..., None, None]
        )

        # Note: before implementing Gauss Newton below,
        # affine_inv_gradient_reduction as defined below was the previous
        # returned value for the affine_inv_gradient.
        # For 3D case, this has shape (3,4).
        affine_inv_gradient_reduction = np.sum(
            affine_inv_gradient, tuple(range(self.moving_image.ndim))
        )

        # Reshape to a single vector. For a 3D case this becomes shape (12,).
        affine_inv_gradient_reduction = affine_inv_gradient_reduction.ravel()

        # For a 3D case, matching_affine_inv_gradient has shape (x,y,z,3,4).
        # For a 3D case, affine_inv_hessian_approx is
        # matching_affine_inv_gradient reshaped to shape (x,y,z,12,1),
        # then matrix multiplied by itself transposed on the last two
        # dimensions, then summed over the spatial dimensions to resultant
        # shape (12,12).
        affine_inv_hessian_approx = matching_affine_inv_gradient * (
            (
                contrast_map_prime
                * np.sqrt(self.matching_weights)
                / self.sigma_matching
            )[..., None, None]
        )
        affine_inv_hessian_approx = affine_inv_hessian_approx.reshape(
            *matching_affine_inv_gradient.shape[:-2], -1, 1
        )
        affine_inv_hessian_approx = (
            affine_inv_hessian_approx
            @ affine_inv_hessian_approx.reshape(
                *affine_inv_hessian_approx.shape[:-2], 1, -1
            )
        )
        affine_inv_hessian_approx = np.sum(
            affine_inv_hessian_approx, tuple(range(self.moving_image.ndim))
        )

        # Solve for affine_inv_gradient.
        try:
            affine_inv_gradient = solve(
                affine_inv_hessian_approx,
                affine_inv_gradient_reduction,
                assume_a="pos",
            ).reshape(matching_affine_inv_gradient.shape[-2:])
        except np.linalg.LinAlgError as exception:
            raise RuntimeError(
                "The Hessian was not invertible in the Gauss-Newton update of"
                "the affine transform. This may be because the image was "
                "constant along one or more dimensions. Consider removing "
                "any constant dimensions. Otherwise you may try using a "
                "smaller value for affine_stepsize, a smaller value for "
                "deformative_stepsize, or a larger value for "
                "sigma_regularization. The values output in Diagnostics may "
                "be of use in determining optimal parameter values."
            ) from exception
        # Append a row of zeros at the end of the 0th dimension.
        zeros = np.zeros((1, self.moving_image.ndim + 1),
                         dtype=self.float_dtype)
        affine_inv_gradient = np.concatenate((affine_inv_gradient, zeros), 0)

        return affine_inv_gradient

    def _update_affine(self, affine_inv_gradient, iteration):
        """
        Update self.affine based on affine_inv_gradient.

        If iteration < self.num_rigid_affine_iterations, project self.affine
        to a rigid affine.

        If self.fixed_affine_scale is provided, it is imposed on self.affine.

        Appends the current self.affine to self.affines.

        Accesses attributes:
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
            self.affine[:-1, :-1] = (
                U
                @ np.diag([self.fixed_affine_scale] * (len(self.affine) - 1))
                @ Vh
            )
        # If self.fixed_affine_scale was not provided (is None), project
        # self.affine to a rigid affine if appropriate.
        elif iteration < self.num_rigid_affine_iterations:
            U, _, Vh = svd(self.affine[:-1, :-1])
            self.affine[:-1, :-1] = U @ Vh

        # Save affine for diagnostics.
        self.affines.append(self.affine)

    def _compute_velocity_fields_gradients(self):
        """
        Compute and return the gradients of the self.velocity_fields.

        Accesses attributes:
            reference_image
            moving_image
            reference_image_axes
            moving_image_axes
            reference_image_coords
            reference_image_spacing
            deformed_reference_image_to_time
            deformed_reference_image
            contrast_deformed_reference_image
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

        matching_error_prime = (
            (self.contrast_deformed_reference_image - self.moving_image)
            * self.matching_weights
            / self.sigma_matching ** 2
        )
        contrast_map_prime = np.zeros_like(self.moving_image, self.float_dtype)
        for power in range(1, self.contrast_order + 1):
            contrast_map_prime += (
                power
                * self.deformed_reference_image ** (power - 1)
                * self.contrast_coefficients[..., power]
            )
        d_matching_d_deformed_reference_image = (
            matching_error_prime * contrast_map_prime
        )
        d_matching_d_deformed_reference_image_padded = np.pad(
            d_matching_d_deformed_reference_image,
            2,
            mode="constant",
            constant_values=0,
        )

        # Set self.phi to identity. self.phi is secretly phi_1t_inv but at the
        # end of the loop it will be phi_10_inv = phi_01 = phi.
        self.phi = np.copy(self.reference_image_coords)

        # Loop backwards across time.
        d_matching_d_velocities = []
        for timestep in reversed(range(self.num_timesteps)):

            # Update phi.
            sample_coords = (
                self.reference_image_coords
                + self.velocity_fields[..., timestep, :] * self.delta_t
            )
            for i in range(self.reference_image.ndim):
                self.phi[..., i] = ndi.map_coordinates(
                    self.phi[..., i] - self.reference_image_coords[..., i],
                    self._normalize_coords(sample_coords),
                    order=1,
                    mode='nearest',
                    output=self.float_dtype,
                ) + sample_coords[..., i]

            # Apply affine by multiplication.
            # This transforms error in the moving_image space back to time t.
            self.affine_phi = _multiply_coords_by_affine(
                self.affine,
                self.phi,
            )

            # Compute the determinant of the gradient of self.phi.
            grad_phi = np.stack(
                np.gradient(
                    self.phi,
                    *self.reference_image_spacing,
                    axis=tuple(range(self.reference_image.ndim)),
                ),
                -1,
            )
            det_grad_phi = _compute_tail_determinant(grad_phi)

            _coords = _compute_axes(
                d_matching_d_deformed_reference_image_padded.shape,
                self.moving_image_spacing,
                dtype=self.float_dtype,
            )
            # Normalize coordinates relative to spacing of 1 for
            # d_matching_d_deformed_reference_image_padded
            coords = [
                ((self.affine_phi[..., i] - _coords[i][0]) /
                 self.moving_image_spacing[i])
                for i in range(self.reference_image.ndim)]
            error_at_t = ndi.map_coordinates(
                d_matching_d_deformed_reference_image_padded,
                coords,
                order=1,
                mode='nearest',
                output=self.float_dtype,
            )

            # The gradient of the reference_image image deformed to time t.
            deformed_reference_image_to_time_gradient = np.stack(
                np.gradient(
                    self.deformed_reference_image_to_time[timestep],
                    *self.reference_image_spacing,
                    axis=tuple(range(self.reference_image.ndim)),
                ),
                -1,
            )

            # The derivative of the matching cost with respect to the velocity
            # at time t is the product of
            # (the error deformed to time t),
            # (the reference_image gradient deformed to time t),
            # & (the determinant of the jacobian of the transformation).
            d_matching_d_velocity_at_t = (
                np.expand_dims(error_at_t * det_grad_phi, -1)
                * deformed_reference_image_to_time_gradient
                * (-1.0)
                * det(self.affine)
            )

            # To convert from derivative to gradient we smooth by applying a
            # physical-unit low-pass filter in the frequency domain.
            matching_cost_at_t_gradient = (
                fftmodule.fftn(
                    d_matching_d_velocity_at_t,
                    axes=tuple(range(self.reference_image.ndim)),
                )
                * np.expand_dims(self.low_pass_filter, -1)
            )
            # Add the gradient of the regularization term.
            matching_cost_at_t_gradient += (
                fftmodule.fftn(
                    self.velocity_fields[..., timestep, :],
                    axes=tuple(range(self.reference_image.ndim)),
                )
                / self.sigma_regularization ** 2
            )
            # Multiply by a voxel-unit low-pass filter to further smooth.
            matching_cost_at_t_gradient *= np.expand_dims(
                self.preconditioner_low_pass_filter, -1
            )
            # Invert fourier transform back to the spatial domain.
            d_matching_d_velocity_at_t = fftmodule.ifftn(
                matching_cost_at_t_gradient,
                axes=tuple(range(self.reference_image.ndim)),
            ).real

            # Naturally this should be inserting at the front because we are
            # looping backwards through time. However, for efficiency we append
            # and reverse the list after the loop.
            d_matching_d_velocities.append(d_matching_d_velocity_at_t)
        d_matching_d_velocities.reverse()

        return d_matching_d_velocities

    def _update_velocity_fields(self, velocity_fields_gradients):
        """
        Update self.velocity_fields based on velocity_fields_gradient.

        Calculates and appends the maximum velocity to
        self.maximum_velocities.

        Accesses attributes:
            deformative_stepsize
            num_timesteps
            velocity_fields
            maximum_velocities

        Updates attributes:
            velocity_fields
            maximum_velocities
        """

        for timestep in range(self.num_timesteps):
            velocity_fields_update = (
                velocity_fields_gradients[timestep] * self.deformative_stepsize
            )
            # Apply a sigmoid squashing function to the velocity_fields_update
            # to ensure they yield an update of less than
            # self.maximum_velocity_fields_update voxels while remaining
            # smooth.
            velocity_fields_update_norm = np.sqrt(
                np.sum(velocity_fields_update ** 2, axis=-1)
            )
            # When the norm is 0 the update is zero so we can change the norm
            # to 1 and avoid division by 0.
            velocity_fields_update_norm[velocity_fields_update_norm == 0] = 1
            velocity_fields_update = (
                velocity_fields_update
                / velocity_fields_update_norm[..., None]
                * np.arctan(
                    velocity_fields_update_norm[..., None]
                    * np.pi
                    / 2
                    / self.maximum_velocity_fields_update
                )
                * self.maximum_velocity_fields_update
                * 2
                / np.pi
            )

            self.velocity_fields[..., timestep, :] -= velocity_fields_update

        # Compute and save maximum velocity for diagnostics.
        maximum_velocity = np.sqrt(
            np.sum(self.velocity_fields ** 2, axis=-1)
        ).max()
        self.maximum_velocities.append(maximum_velocity)

    def _compute_affine_phi(self):
        """
        Compute and set self.affine_phi. Called once in case there were no
        deformative iterations to set it.

        Accesses attributes:
            reference_image_axes
            reference_image_coords
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

        # Set self.phi to identity. self.phi is secretly phi_1t_inv but at the
        # end of the loop it will be phi_10_inv = phi_01 = phi.
        self.phi = np.copy(self.reference_image_coords)

        # Loop backwards across time.
        for timestep in reversed(range(self.num_timesteps)):

            # Update phi.
            sample_coords = (
                self.reference_image_coords
                + self.velocity_fields[..., timestep, :] * self.delta_t
            )
            for i in range(self.reference_image.ndim):
                self.phi[..., i] = ndi.map_coordinates(
                    self.phi[..., i] - self.reference_image_coords[..., i],
                    self._normalize_coords(sample_coords),
                    order=1,
                    mode='nearest',
                    output=self.float_dtype,
                ) + sample_coords[..., i]


            # Apply affine by multiplication.
            # This transforms error in the moving_image space back to time t.
            self.affine_phi = _multiply_coords_by_affine(self.affine, self.phi)

    # End _Lddmm.


def diffeomorphic_metric_mapping(
    reference_image,
    moving_image,
    reference_image_spacing=None,
    moving_image_spacing=None,
    deformative_stepsize=None,
    sigma_regularization=None,
    contrast_order=None,
    **kwargs,
):
    """
    Compute a registration between grayscale images reference_image and
    moving_image, to be applied with scipy.ndimage.map_coordinates.

    Parameters
    ----------
    reference_image: np.ndarray
        The ideally clean reference_image image being registered to the
        moving_image.
    moving_image: np.ndarray
        The potentially messier moving_image image being registered to.
    reference_image_spacing: float, seq, optional
        A scalar or list of scalars indicating the spacing of the
        reference_image. Overrides 0 input. By default 1.
    moving_image_spacing: float, seq, optional
        A scalar or list of scalars indicating the spacing of the
        moving_image. Overrides 0 input. By default 1.
    deformative_stepsize: float, optional
        The stepsize for deformative adjustments. Optimal values are
        problem-specific. Setting preconditioner_velocity_smooth_length
        increases the appropriate value of deformative_stepsize.
        If equal to 0 then the result is affine-only registration.
        By default 0.
    sigma_regularization: float, optional
        A scalar indicating the freedom to deform. Small values put
        harsher constraints on the smoothness of a deformation.
        With sufficiently large values, the registration will overfit any
        noise in the moving_image, leading to unrealistic deformations.
        However, this may still be appropriate with a small
        num_iterations.
        Note that if deformative_stepsize / sigma_regularization**2 is not
        much less than 1, an error may occur.
        Overrides 0 input. By default np.inf.
    contrast_order: int, optional
        The order of the polynomial fit between the contrasts of the
        reference_image and moving_image. This is important to set greater
        than 1 if reference_image and moving_image are cross-modal.
        3 is generally good for histology. Must be at least 1.
        By default 1.
    **kwargs
        The above parameters are sufficient for the majority of
        registrations; however, some require additional fine-tuning or
        modification. A number of additional keyword arguments are
        accessible via kwargs to provide a rich environment of options for
        tailoring a particular registration.

        Among these options are
        parameters whose default values may be overridden such as
        num_iterations, and boolean flags that activate additional
        features including spatially_varying_contrast_map and
        artifact_and_background_classification. If set to True, these
        features are further parametrized by other kwargs.

        It is important to note that some kwarg specifications will affect
        the validity of previously calibrated values for other parameters,
        most notably deformative_stepsize. Options include:

            multiscales: float, seq, optional
                A scalar, list of scalars, or list of lists or np.ndarray
                of scalars, determining the levels of downsampling at
                which the registration should be performed before moving
                on to the next.

                Values must be either all at least 1, or all at most 1.
                Both options are interpreted as downsampling. For example,
                multiscales=[10, 3, 1] will result in the reference_image
                and moving_image being downsampled by a factor of 10 and
                registered. This registration will be upsampled and used
                to initialize another registration of the reference_image
                and moving_image downsampled by 3, and then again on the
                undownsampled data (downsampled by 1).
                multiscales=[1/10, 1/3, 1] is equivalent.

                Further, the scale for each dimension can be specified,
                e.g. multiscales=[ [10, 5, 5], [3, 3, 3], 1] for a 3D
                registration will result in the reference_image and
                moving_image downsampled by [10, 5, 5], then [3, 3, 3],
                then [1, 1, 1].

                If multiscales is provided with more than 1 value, all
                other arguments with the exceptions of reference_image,
                moving_image, reference_image_spacing,
                moving_image_spacing, initial_affine,
                initial_velocity_fields, and
                initial_contrast_coefficients, which may be provided for
                the first value in multiscales, may optionally be provided
                as sequences with length equal to the number of values
                provided to multiscales. Each such value is used at the
                corresponding scale.

                reference_image_spacing and moving_image_spacing are given
                once to indicate the spacing of the reference_image and
                moving_image as provided, but will be internally adjusted
                at each scale.

                multiscales should be provided as descending values.
                By default 1.
            num_iterations: int, optional
                The total number of iterations. By default 300.
            num_affine_only_iterations: int, optional
                The number of iterations at the start of the process
                without deformative adjustments. By default 100.
            num_rigid_affine_iterations: int, optional
                The number of iterations at the start of the process in
                which the affine is kept rigid. By default 50.
            affine_stepsize: float, optional
                The unitless stepsize for affine adjustments. Should be
                between 0 and 1. By default 0.3.
            fixed_affine_scale: float, optional
                The scale to impose on the affine at all iterations. If
                None, no scale is imposed. Otherwise, this has the effect
                of making the affine always rigid. By default None.
            velocity_smooth_length: float, optional
                The length scale of smoothing of the velocity_fields in
                physical units. Affects the optimum velocity_fields
                smoothness.
                By default 2 * np.max(self.reference_image_spacing).
            preconditioner_velocity_smooth_length: float, optional
                The length of preconditioner smoothing of the
                velocity_fields in physical units. Affects the
                optimization of the velocity_fields, but not the optimum.
                By default 0.
            maximum_velocity_fields_update: float, optional
                The maximum allowed update to the velocity_fields in
                physical units. Affects the optimization of the
                velocity_fields, but not the optimum. Overrides 0 input.
                By default np.max(self.reference_image.shape
                * self.reference_image_spacing).
            num_timesteps: int, optional
                The number of composed sub-transformations in the
                diffeomorphism. Overrides 0 input. By default 5.
            spatially_varying_contrast_map: bool, optional
                If True, uses a polynomial per voxel to compute the
                contrast map rather than a single polynomial.
                By default False.
            contrast_iterations: int, optional
                The number of iterations of gradient descent to converge
                toward the optimal contrast_coefficients if
                spatially_varying_contrast_map == True. Overrides 0 input.
                By default 5.
            sigma_contrast: float, optional
                The scale of variation in the contrast_coefficients if
                spatially_varying_contrast_map == True. Overrides 0 input.
                By default 1.
            contrast_smooth_length: float, optional
                The length scale of smoothing of the contrast_coefficients
                if spatially_varying_contrast_map == True.
                Overrides 0 input.
                By default 2 * np.max(self.moving_image_spacing).
            sigma_matching: float, optional
                An estimate of the spread of the noise in the moving_image,
                representing the tradeoff between the regularity and
                accuracy of the registration, where a smaller value should
                result in a less smooth, more accurate result. Typically it
                should be set to an estimate of the standard deviation of
                the noise in the image, particularly with artifacts.
                Overrides 0 input.
                By default the standard deviation of the moving_image.
            artifact_and_background_classification: bool, optional
                If True, artifacts and background are jointly classified
                with registration using sigma_artifact, artifact_prior,
                sigma_background, and background_prior.
                Artifacts refer to excessively bright voxels while
                background refers to excessively dim voxels.
                By default False.
            sigma_artifact: float, optional
                The level of expected variation between artifact and
                non-artifact intensities. Overrides 0 input.
                By default 5 * sigma_matching.
            sigma_background: float, optional
                The level of expected variation between background and
                non-background intensities. Overrides 0 input.
                By default 2 * sigma_matching.
            artifact_prior: float, optional
                The prior probability at which we expect to find that any
                given voxel is artifact. By default 1/3.
            background_prior: float, optional
                The prior probability at which we expect to find that any
                given voxel is background. By default 1/3.
            initial_affine: np.ndarray, optional
                The affine array that the registration will begin with.
                By default np.eye(reference_image.ndim + 1).
            initial_contrast_coefficients: np.ndarray, optional
                The contrast coefficients that the registration will begin
                with. If None, the 0th order coefficient(s) are set to
                np.mean(self.moving_image) - np.mean(self.reference_image)
                * np.std(self.moving_image) / np.std(self.reference_image),
                if self.contrast_order > 1, the 1st order coefficient(s)
                are set to
                np.std(self.moving_image) / np.std(self.reference_image),
                and all others are set to zero. By default None.
            initial_velocity_fields: np.ndarray, optional
                The velocity fields that the registration will begin with.
                By default all zeros.
            map_coordinates_ify: bool, optional
                If True, the position fields encoding the transformation
                will be converted to units of voxels in the expected format
                of scipy.ndimage.map_coordinates.
                If False, they are left centered and in physical units with
                the exising in the last dimension. By default True.

    Returns
    -------
    namedtuple
        A namedtuple object containing 4 elements:

            moving_image_to_reference_image_transform: ndarray
                The position-field for transforming an image from the
                moving_image-space to the reference_image-space.
            reference_image_to_moving_image_transform: ndarray
                The position-field for transforming an image from the
                reference_image-space to the moving_image-space.
            internals: namedtuple
                A namedtuple object containing internal components of the
                above transforms, or ultimate position-fields, that may be of
                interest.

                    affine: ndarray
                        The affine component of
                        reference_image_to_moving_image_transform in
                        homogenous coordinates.
                    contrast_coefficients: ndarray
                        The weights for the polynomial used to map the
                        contrast from the reference_image to the moving_image.
                    velocity_fields: ndarray
                        The flow-fields whose integral across time is the
                        deformative component of the ultimate position-field
                        transform.
                    reference_image_deformation: ndarray
                        The deformative component of
                        reference_image_to_moving_image_transform.
                    reference_image_deformation_inverse: ndarray
                        The inverse of reference_image_deformation and the
                        deformative component of
                        moving_image_to_reference_image_transform.

            diagnostics: namedtuple
                A namedtuple containing values accumulated across all
                iterations including across multiple scales, highly useful
                for visualizing the progression of the registration and
                calibrating parameters.

                    affines: list
                        The affine arrays computed at each iteration.
                    maximum_velocities: list
                        The greatest norm of velocity_fields at each
                        iteration.
                    matching_energies: list
                        The sum of square error penalty at each iteration.
                    regularization_energies: list
                        The roughness penalty on velocity_fields at each
                        iteration.
                    total_energies: list
                        The sum of matching_energy and regularization_energy,
                        used as the overall cost function, at each iteration.

    Raises
    ------
    ValueError
        Raised if unrecognized keyword arguments are provided.
    ValueError
        Raised if multiscales is provided with values both above and below 1.

    References
    ----------
    .. [1] Tward, Daniel J. et al. "Diffeomorphic registration with intensity
        transformation and missing data: Application to 3D digital pathology of
        Alzheimer's disease." Frontiers in neuroscience 14 (2020).
        :DOI:`10.3389/fnins.2020.00052`
    .. [2] Beg, M. Faisal, et al. "Computing large deformation metric mappings
        via geodesic flows of diffeomorphisms." International journal of
        computer vision 61.2 (2005): 139-157.
        :DOI:`10.1023/B:VISI.0000043755.93987.aa`

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.ndimage import rotate
    >>> from skimage.registration import diffeomorphic_metric_mapping
    >>> from scipy.ndimage import map_coordinates
    >>> #
    >>> # Define images.
    >>> # The reference_image is registered to the moving_image image
    >>> # but both transformations are returned.
    >>> #
    >>> # reference_image is a binary ellipse with semi-radii 5 and 8 in
    >>> # dimensions 0 and 1. The overall shape is (19, 25).
    >>> # moving_image is a 30 degree rotation of reference_image.
    >>> #
    >>> reference_image = np.array([[(col-12)**2/8**2 + (row-9)**2/5**2
    ... <= 1 for col in range(25)] for row in range(19)], float)
    >>> moving_image = rotate(reference_image, 30)
    >>> #
    >>> # Register the reference_image to the moving_image,
    >>> # then deform the reference_image and moving_image
    >>> # to match the other.
    >>> #
    >>> lddmm_output = diffeomorphic_metric_mapping(
    ... reference_image, moving_image, deformative_stepsize=0.5)
    >>> #
    >>> deformed_moving_image = map_coordinates(moving_image,
    ... lddmm_output.moving_image_to_reference_image_transform)
    >>> #
    >>> deformed_reference_image = map_coordinates(reference_image,
    ... lddmm_output.reference_image_to_moving_image_transform)

    """

    # Validate kwargs.
    recognized_kwarg_keys = {
        "multiscales",
        "num_iterations",
        "num_affine_only_iterations",
        "num_rigid_affine_iterations",
        "affine_stepsize",
        "fixed_affine_scale",
        "velocity_smooth_length",
        "preconditioner_velocity_smooth_length",
        "maximum_velocity_fields_update",
        "num_timesteps",
        "spatially_varying_contrast_map",
        "contrast_iterations",
        "sigma_contrast",
        "contrast_smooth_length",
        "sigma_matching",
        "artifact_and_background_classification",
        "sigma_artifact",
        "sigma_background",
        "artifact_prior",
        "background_prior",
        "initial_affine",
        "initial_contrast_coefficients",
        "initial_velocity_fields",
        "map_coordinates_ify",
    }
    invalid_kwargs = list(
        filter(
            lambda kwarg_key: kwarg_key not in recognized_kwarg_keys,
            kwargs.keys(),
        )
    )
    if invalid_kwargs:
        raise ValueError(
            "One or more unexpected keyword arguments were encountered.\n"
            f"Unrecognized keyword arguments: {invalid_kwargs}."
        )
    # Augment kwargs with None entries for unspecified arguments.
    for kwarg_key in recognized_kwarg_keys:
        if kwarg_key not in kwargs.keys():
            kwargs[kwarg_key] = None

    # Validate images and spacings.
    # Images.
    float_dtype = np.float32 if reference_image.dtype == np.float32 else np.float64
    reference_image = _validate_ndarray(reference_image, dtype=float_dtype)
    moving_image = _validate_ndarray(
        moving_image, dtype=float_dtype, required_ndim=reference_image.ndim
    )
    # spacing.
    reference_image_spacing = _validate_scalar_to_multi(
        reference_image_spacing if reference_image_spacing is not None else 1,
        reference_image.ndim,
        float_dtype,
    )
    moving_image_spacing = _validate_scalar_to_multi(
        moving_image_spacing if moving_image_spacing is not None else 1,
        moving_image.ndim,
        float_dtype,
    )

    # Unpack multiscale-relevant kwargs.
    multiscales = kwargs["multiscales"]
    initial_affine = kwargs["initial_affine"]
    initial_contrast_coefficients = kwargs["initial_contrast_coefficients"]
    initial_velocity_fields = kwargs["initial_velocity_fields"]

    # Validate multiscales.
    # Note: aside from map_coordinates_ify,
    # this is the only argument not passed to _Lddmm.
    if multiscales is None:
        multiscales = 1
    multiscales = list(np.atleast_1d(np.array(multiscales, dtype=object)))
    # multiscales is a list.
    for index, scale in enumerate(multiscales):
        multiscales[index] = _validate_scalar_to_multi(
            scale, size=reference_image.ndim, dtype=float
        )
    multiscales = _validate_ndarray(
        multiscales, required_shape=(-1, reference_image.ndim)
    )
    # Each scale in multiscales has length reference_image.ndim.
    if np.all(multiscales >= 1):
        multiscales = 1 / multiscales
    elif not np.all(multiscales <= 1):
        raise ValueError(
            "If provided, the values in multiscales must be either all >= 1 "
            "or all <= 1."
        )
    # All values in multiscales are <= 1. If provided with all scales greater
    # than or equal to 1, multiscales are ingested as their reciprocals.

    # Validate potential multiscale arguments.

    # multiscale_lddmm_kwargs contains all kwargs except:
    # reference_image, moving_image,
    # reference_image_spacing, moving_image_spacing,
    # multiscales,
    # initial_affine, initial_contrast_coefficients, initial_velocity_fields,
    # and map_coordinates_ify.
    multiscale_lddmm_kwarg_exclusions = {
        "multiscales",
        "initial_affine",
        "initial_contrast_coefficients",
        "initial_velocity_fields",
        "map_coordinates_ify",
    }
    multiscale_lddmm_kwargs = {
        kwarg_name: kwarg_value for kwarg_name, kwarg_value in kwargs.items()
        if kwarg_name not in multiscale_lddmm_kwarg_exclusions
    }
    # Include parameters that are not in the kwargs dictionary.
    multiscale_lddmm_kwargs.update(
        deformative_stepsize=deformative_stepsize,
        sigma_regularization=sigma_regularization,
        contrast_order=contrast_order,
    )
    # All multiscale_lddmm_kwargs should be used in _Lddmm as scalars, not
    # sequences. Here, they are made into sequences corresponding to the
    # length of multiscales.
    for kwarg_name, kwarg_value in multiscale_lddmm_kwargs.items():
        multiscale_lddmm_kwargs[kwarg_name] = _validate_scalar_to_multi(
            kwarg_value,
            size=len(multiscales),
            dtype=None,
            reject_nans=False,
        )
    # Each value in the multiscale_lddmm_kwargs dictionary is an array with
    # shape (len(multiscales),).

    # Initialize diagnostic accumulators to None.
    affines = None
    maximum_velocities = None
    matching_energies = None
    regularization_energies = None
    total_energies = None

    for scale_index, scale in enumerate(multiscales):

        # Extract appropriate multiscale_lddmm_kwargs.
        this_scale_lddmm_kwargs = {
            kwarg_name: multiscale_lddmm_kwargs[kwarg_name][scale_index]
            for kwarg_name in multiscale_lddmm_kwargs
        }

        # rescale images and spacings.
        # reference_image.
        reference_image_scale = (
            np.round(scale * reference_image.shape) / reference_image.shape
        ).astype(float_dtype, copy=False)
        scaled_reference_image = rescale(
            reference_image, reference_image_scale
        )
        scaled_reference_image_spacing = (
            reference_image_spacing / reference_image_scale
        )
        # moving_image.
        moving_image_scale = (
            np.round(scale * moving_image.shape) / moving_image.shape
        ).astype(float_dtype, copy=False)
        scaled_moving_image = rescale(moving_image, moving_image_scale)
        scaled_moving_image_spacing = moving_image_spacing / moving_image_scale

        # Collect non-multiscale_lddmm_kwargs
        # Note: user arguments initial_affine, initial_contrast_coefficients,
        # and initial_velocity_fields are overwritten in this loop.
        multiscale_exempt_lddmm_kwargs = dict(
            # Images.
            reference_image=scaled_reference_image,
            moving_image=scaled_moving_image,
            # Image spacings.
            reference_image_spacing=scaled_reference_image_spacing,
            moving_image_spacing=scaled_moving_image_spacing,
            # Initial values.
            initial_affine=initial_affine,
            initial_contrast_coefficients=initial_contrast_coefficients,
            initial_velocity_fields=initial_velocity_fields,
            # Diagnostic accumulators.
            affines=affines,
            maximum_velocities=maximum_velocities,
            matching_energies=matching_energies,
            regularization_energies=regularization_energies,
            total_energies=total_energies,
        )

        # Perform registration.

        # Set up _Lddmm instance.
        lddmm = _Lddmm(
            **this_scale_lddmm_kwargs, **multiscale_exempt_lddmm_kwargs
        )

        lddmm.register()

        # Overwrite initials for next scale if applicable.
        if scale_index < len(multiscales) - 1:
            # Initial diagnostic accumulators.
            affines = lddmm.affines
            maximum_velocities = lddmm.maximum_velocities
            matching_energies = lddmm.matching_energies
            regularization_energies = lddmm.regularization_energies
            total_energies = lddmm.total_energies
            # initial_affine.
            initial_affine = lddmm.affine
            # initial_contrast_coefficients.
            if (
                multiscale_lddmm_kwargs["spatially_varying_contrast_map"][
                    scale_index + 1
                ]
                and multiscale_lddmm_kwargs["spatially_varying_contrast_map"][
                    scale_index
                ]
            ):
                # If spatially_varying_contrast_map at next scale and at this
                # scale, resize contrast_coefficients.
                next_moving_image_shape = np.round(
                    multiscales[scale_index + 1] * moving_image.shape
                )
                initial_contrast_coefficients = resize(
                    lddmm.contrast_coefficients,
                    (
                        *next_moving_image_shape,
                        multiscale_lddmm_kwargs["contrast_order"][
                            scale_index + 1
                        ]
                        + 1,
                    ),
                )
            elif (
                not multiscale_lddmm_kwargs["spatially_varying_contrast_map"][
                    scale_index + 1
                ]
                and multiscale_lddmm_kwargs["spatially_varying_contrast_map"][
                    scale_index
                ]
            ):
                # If spatially_varying_contrast_map at this scale but not at
                # next scale, average contrast_coefficients.
                initial_contrast_coefficients = np.mean(
                    lddmm.contrast_coefficients,
                    axis=np.arange(reference_image.ndim),
                )
            else:
                # If spatially_varying_contrast_map at next scale but not this
                # scale or at neither scale, initialize directly.
                initial_contrast_coefficients = lddmm.contrast_coefficients
            # initial_velocity_fields.
            next_reference_image_shape = np.round(
                multiscales[scale_index + 1] * reference_image.shape
            )
            initial_velocity_fields = sinc_resample(
                lddmm.velocity_fields,
                new_shape=(
                    *next_reference_image_shape,
                    multiscale_lddmm_kwargs["num_timesteps"][scale_index + 1]
                    or lddmm.num_timesteps,
                    reference_image.ndim,
                ),
            )
        # End multiscales loop.

    # If map_coordinates_ify, convert centered, physical-space position-fields
    # to voxel-space position-fields.
    if kwargs["map_coordinates_ify"] is None or kwargs["map_coordinates_ify"]:
        # resize to match the shape of the appropriate image,
        # subtract the identity coordinate vector at spatial indices 0,
        # (assuming centered coordinates)
        # divide by the original spacing of the image,
        # and move the coordinate axis to the front.
        lddmm.phi = np.moveaxis(
            (
                resize(
                    lddmm.phi, (*reference_image.shape, reference_image.ndim)
                )
                - (
                    -np.subtract(reference_image.shape, 1)
                    / 2
                    * reference_image_spacing
                )
            )
            / reference_image_spacing,
            -1,
            0,
        )
        lddmm.phi_inv = np.moveaxis(
            (
                resize(
                    lddmm.phi_inv,
                    (*reference_image.shape, reference_image.ndim),
                )
                - (
                    -np.subtract(reference_image.shape, 1)
                    / 2
                    * reference_image_spacing
                )
            )
            / reference_image_spacing,
            -1,
            0,
        )
        lddmm.affine_phi = np.moveaxis(
            (
                resize(
                    lddmm.affine_phi,
                    (*reference_image.shape, reference_image.ndim),
                )
                - (
                    -np.subtract(moving_image.shape, 1)
                    / 2
                    * moving_image_spacing
                )
            )
            / moving_image_spacing,
            -1,
            0,
        )
        lddmm.phi_inv_affine_inv = np.moveaxis(
            (
                resize(
                    lddmm.phi_inv_affine_inv,
                    (*moving_image.shape, reference_image.ndim),
                )
                - (
                    -np.subtract(reference_image.shape, 1)
                    / 2
                    * reference_image_spacing
                )
            )
            / reference_image_spacing,
            -1,
            0,
        )

    # Define namedtuple objects for lddmm_output.
    Lddmm_output = namedtuple(
        "Lddmm_output",
        (
            "moving_image_to_reference_image_transform",
            "reference_image_to_moving_image_transform",
            "internals",
            "diagnostics",
        ),
    )
    Internals = namedtuple(
        "Internals",
        (
            "affine",
            "contrast_coefficients",
            "velocity_fields",
            "reference_image_deformation",
            "reference_image_deformation_inverse",
        ),
    )
    Diagnostics = namedtuple(
        "Diagnostics",
        (
            "affines",
            "maximum_velocities",
            "matching_energies",
            "regularization_energies",
            "total_energies",
        ),
    )

    # Construct lddmm_output.
    internals = Internals(
        lddmm.affine,
        lddmm.contrast_coefficients,
        lddmm.velocity_fields,
        lddmm.phi,
        lddmm.phi_inv,
    )
    diagnostics = Diagnostics(
        lddmm.affines,
        lddmm.maximum_velocities,
        lddmm.matching_energies,
        lddmm.regularization_energies,
        lddmm.total_energies,
    )
    lddmm_output = Lddmm_output(
        lddmm.affine_phi,
        lddmm.phi_inv_affine_inv,
        internals,
        diagnostics,
    )

    return lddmm_output
