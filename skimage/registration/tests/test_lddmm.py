import pytest
from itertools import product

import numpy as np
from scipy.linalg import inv
from scipy.ndimage import rotate
from scipy.ndimage import map_coordinates

from skimage.registration import _lddmm_utilities

from skimage.registration._lddmm import lddmm_register

"""
Test lddmm_register.
"""

class Test_lddmm_register:

    def _test_lddmm_register(self, rtol=0, atol=1-1e-9, **lddmm_register_kwargs):
        """A helper method for this class to verify registrations once they are computed."""

        lddmm_output = lddmm_register(**lddmm_register_kwargs)

        template = lddmm_register_kwargs['template'].astype(float)
        target = lddmm_register_kwargs['target'].astype(float)
        template_resolution = lddmm_register_kwargs['template_resolution'] if 'template_resolution' in lddmm_register_kwargs.keys() else 1
        target_resolution = lddmm_register_kwargs['target_resolution'] if 'target_resolution' in lddmm_register_kwargs.keys() else 1

        # Applying the transforms using map_coordinates assumes map_coordinates_ify was left as True.

        deformed_target = map_coordinates(
            input=target,
            coordinates=lddmm_output.target_to_template_transform,
        )

        deformed_template = map_coordinates(
            input=template,
            coordinates=lddmm_output.template_to_target_transform,
        )

        assert np.allclose(deformed_template, target, rtol=rtol, atol=atol)
        assert np.allclose(deformed_target, template, rtol=rtol, atol=atol)

    def test_2D_identity_registration(self):

        ndims = 2
        radius = 3
        zero_space = 2
        center_pixel = True
        template = np.zeros(tuple([(radius + zero_space) * 2 + center_pixel] * ndims))
        for indices in product(*map(range, template.shape)):
            indices = np.array(indices)
            if np.sqrt(np.sum((indices - (radius + zero_space) + (not center_pixel) / 2)**2)) <= radius:
                template[tuple(indices)] = 1
        target = np.copy(template)

        lddmm_register_kwargs = dict(
            template=template,
            target=target,
            num_iterations=1,
        )

        self._test_lddmm_register(**lddmm_register_kwargs)

    def test_3D_identity_registration(self):

        ndims = 3
        radius = 3
        zero_space = 2
        center_pixel = True
        template = np.zeros(tuple([(radius + zero_space) * 2 + center_pixel] * ndims))
        for indices in product(*map(range, template.shape)):
            indices = np.array(indices)
            if np.sqrt(np.sum((indices - (radius + zero_space) + (not center_pixel) / 2)**2)) <= radius:
                template[tuple(indices)] = 1
        target = np.copy(template)

        lddmm_register_kwargs = dict(
            template=template,
            target=target,
            num_iterations=1,
        )

        self._test_lddmm_register(**lddmm_register_kwargs)

    def test_4D_identity_registration(self):

        ndims = 4
        radius = 3
        zero_space = 2
        center_pixel = True
        template = np.zeros(tuple([(radius + zero_space) * 2 + center_pixel] * ndims))
        for indices in product(*map(range, template.shape)):
            indices = np.array(indices)
            if np.sqrt(np.sum((indices - (radius + zero_space) + (not center_pixel) / 2)**2)) <= radius:
                template[tuple(indices)] = 1
        target = np.copy(template)

        lddmm_register_kwargs = dict(
            template=template,
            target=target,
            num_iterations=1,
        )

        self._test_lddmm_register(**lddmm_register_kwargs)

    def test_identity_disk_to_disk_registration(self):

        template = np.array([[(col-4)**2 + (row-4)**2 <= 4**2 for col in range(9)] for row in range(9)], int)
        target = np.copy(template)

        lddmm_register_kwargs = dict(
            template=template,
            target=target,
            num_affine_only_iterations=0,
        )

        self._test_lddmm_register(**lddmm_register_kwargs)

    def test_rigid_affine_only_ellipsoid_to_ellipsoid_registration(self):

        # template (before padding) has shape (21, 29) and semi-radii 4 and 10.
        template = np.array([[(col-14)**2/10**2 + (row-8)**2/4**2 <= 1 for col in range(29)] for row in range(17)], int)
        # templata and target are opposite rotations of an unrotated ellipsoid for symmetry.
        target = rotate(template, 45/2)
        template = rotate(template, -45/2)

        lddmm_register_kwargs = dict(
            template=template,
            target=target,
            num_iterations=50,
            num_affine_only_iterations=50,
            num_rigid_affine_iterations=50,
        )

        self._test_lddmm_register(**lddmm_register_kwargs)

    def test_non_rigid_affine_only_ellipsoid_to_ellipsoid_registration(self):

        # template (before padding) has shape (21, 29) and semi-radii 6 and 10.
        template = np.array([[(col-14)**2/10**2 + (row-10)**2/6**2 <= 1 for col in range(29)] for row in range(21)], int)
        # target is a rotation of template.
        target = rotate(template, 30)

        lddmm_register_kwargs = dict(
            template=template,
            target=target,
            num_iterations=50,
            num_affine_only_iterations=50,
            num_rigid_affine_iterations=0,
        )

        self._test_lddmm_register(**lddmm_register_kwargs)

    def test_partially_rigid_affine_only_ellipsoid_to_ellipsoid_registration(self):

        # template (before padding) has shape (21, 29) and semi-radii 6 and 10.
        template = np.array([[(col-14)**2/10**2 + (row-10)**2/6**2 <= 1 for col in range(29)] for row in range(21)], int)
        # target is a rotation of template.
        target = rotate(template, 30)

        lddmm_register_kwargs = dict(
            template=template,
            target=target,
            num_iterations=100,
            num_affine_only_iterations=100,
            num_rigid_affine_iterations=50,
        )

        self._test_lddmm_register(**lddmm_register_kwargs)

    def test_deformative_only_disk_to_ellipsoid_registration(self):

        # template has shape (25, 25) and radius 8.
        template = np.array([[(col-12)**2 + (row-12)**2 <= 8**2 for col in range(25)] for row in range(25)], int)
        # target has shape (21, 29) and semi-radii 6 and 10.
        target = np.array([[(col-14)**2/10**2 + (row-10)**2/6**2 <= 1 for col in range(29)] for row in range(21)], int)

        lddmm_register_kwargs = dict(
            template=template,
            target=target,
            num_iterations=150,
            num_affine_only_iterations=0,
            affine_stepsize=0,
            deformative_stepsize=0.5,
        )
        
        self._test_lddmm_register(**lddmm_register_kwargs)

    def test_general_ellipsoid_to_ellipsoid_registration(self):

        # target has shape (21, 29) and semi-radii 6 and 10.
        template = np.array([[(col-14)**2/10**2 + (row-10)**2/6**2 <= 1 for col in range(29)] for row in range(21)], int)
        # target has shape (21, 29) and semi-radii 6 and 10.
        target = rotate(template, 30)

        lddmm_register_kwargs = dict(
            template=template,
            target=target,
            deformative_stepsize=0.5,
        )

        self._test_lddmm_register(**lddmm_register_kwargs)

    def test_identity_multiscale_registration(self):

        # target has shape (21, 29) and semi-radii 6 and 10.
        template = np.array([[(col-14)**2/12**2 + (row-10)**2/8**2 <= 1 for col in range(29)] for row in range(21)], int)
        target = np.copy(template)

        lddmm_register_kwargs = dict(
            template=template,
            target=target,
            num_iterations=1,
            multiscales=[5, (2,3), [3,2], 1],
        )

        self._test_lddmm_register(**lddmm_register_kwargs)


if __name__ == "__main__":
    test_lddmm_register()
