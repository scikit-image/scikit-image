import pytest
from itertools import product

import numpy as np
from scipy.linalg import inv
from scipy.ndimage import rotate

from skimage.registration import _lddmm_utilities

from skimage.registration._lddmm import _generate_position_field
from skimage.registration._lddmm import _apply_position_field
from skimage.registration._lddmm import apply_lddmm
from skimage.registration._lddmm import lddmm_register

"""
Test _generate_position_field.
"""

@pytest.mark.parametrize('deform_to', ['template', 'target'])
def test__generate_position_field(deform_to):

    # Test identity affine and identity velocity_fields.

    num_timesteps = 10

    template_shape = (3,4,5)
    template_resolution = 1
    target_shape = (2,4,6)
    target_resolution = 1
    velocity_fields = np.zeros((*template_shape, num_timesteps, len(template_shape)))
    velocity_field_resolution = 1
    affine = np.eye(4)

    if deform_to == 'template':
        expected_output = _lddmm_utilities._compute_coords(template_shape, template_resolution)
    elif deform_to == 'target':
        expected_output = _lddmm_utilities._compute_coords(target_shape, target_resolution)

    position_field = _generate_position_field(
        affine=affine,
        velocity_fields=velocity_fields,
        velocity_field_resolution=velocity_field_resolution,
        template_shape=template_shape,
        template_resolution=template_resolution,
        target_shape=target_shape,
        target_resolution=target_resolution,
        deform_to=deform_to,
    )

    assert np.array_equal(position_field, expected_output)

    # Test identity affine and constant shift velocity_fields.

    num_timesteps = 10

    template_shape = (3,4,5)
    template_resolution = 1
    target_shape = (2,4,6)
    target_resolution = 1
    velocity_fields = np.ones((*template_shape, num_timesteps, len(template_shape)))
    velocity_field_resolution = 1
    affine = np.eye(4)

    if deform_to == 'template':
        expected_output = _lddmm_utilities._compute_coords(template_shape, template_resolution) + 1
    elif deform_to == 'target':
        expected_output = _lddmm_utilities._compute_coords(target_shape, target_resolution) - 1

    position_field = _generate_position_field(
        affine=affine,
        velocity_fields=velocity_fields,
        velocity_field_resolution=velocity_field_resolution,
        template_shape=template_shape,
        template_resolution=template_resolution,
        target_shape=target_shape,
        target_resolution=target_resolution,
        deform_to=deform_to,
    )

    assert np.allclose(position_field, expected_output)

    # Test rotational affine and identity velocity_fields.

    num_timesteps = 10

    template_shape = (3,4,5)
    template_resolution = 1
    target_shape = (2,4,6)
    target_resolution = 1
    velocity_fields = np.zeros((*template_shape, num_timesteps, len(template_shape)))
    velocity_field_resolution = 1
    # Indicates a 90 degree rotation to the right.
    affine = np.array([
        [0,1,0,0], 
        [-1,0,0,0], 
        [0,0,1,0], 
        [0,0,0,1], 
    ])

    if deform_to == 'template':
        expected_output = _lddmm_utilities._multiply_coords_by_affine(affine, 
            _lddmm_utilities._compute_coords(template_shape, template_resolution))
    elif deform_to == 'target':
        expected_output = _lddmm_utilities._multiply_coords_by_affine(inv(affine), 
            _lddmm_utilities._compute_coords(target_shape, target_resolution))

    position_field = _generate_position_field(
        affine=affine,
        velocity_fields=velocity_fields,
        velocity_field_resolution=velocity_field_resolution,
        template_shape=template_shape,
        template_resolution=template_resolution,
        target_shape=target_shape,
        target_resolution=target_resolution,
        deform_to=deform_to,
    )
    
    assert np.allclose(position_field, expected_output)

"""
Test _apply_position_field.
"""

def test__apply_position_field():
    
    # Test simplest identity position_field.

    subject = np.arange(3*4).reshape(3,4)
    subject_resolution = 1
    output_resolution = 1
    position_field_resolution = subject_resolution
    position_field = _lddmm_utilities._compute_coords(subject.shape, position_field_resolution)

    deformed_subject = _apply_position_field(
        subject=subject,
        subject_resolution=subject_resolution,
        output_resolution=output_resolution,
        position_field=position_field,
        position_field_resolution=position_field_resolution,
    )
    expected_output = subject
    assert np.allclose(deformed_subject, expected_output)

    # Test identity position_field with different output_resolution.

    subject = np.array([
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
    ])
    subject_resolution = 1
    output_resolution = 2
    position_field_resolution = subject_resolution
    position_field = _lddmm_utilities._compute_coords(subject.shape, position_field_resolution)

    deformed_subject = _apply_position_field(
        subject=subject,
        subject_resolution=subject_resolution,
        output_resolution=output_resolution,
        position_field=position_field,
        position_field_resolution=position_field_resolution,
    )
    expected_output = np.array([
        [0,0,0,0],
        [0,1,1,0],
        [0,1,1,0],
        [0,0,0,0],
    ])
    assert np.allclose(deformed_subject, expected_output)
    
    # Test constant shifting position_field with simple extrapolation case.

    # Note: applying a leftward shift to the position_field is done by subtracting 1 from the appropriate dimension.
    # The corresponding effect on the deformed_subject is a shift to the right.

    subject = np.array([
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,1,1,1,1,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
    ])
    subject_resolution = 1
    output_resolution = 1
    position_field_resolution = subject_resolution
    position_field = _lddmm_utilities._compute_coords(subject.shape, position_field_resolution) + [0, -1] # Shift to the left by 1.

    deformed_subject = _apply_position_field(
        subject=subject,
        subject_resolution=subject_resolution,
        output_resolution=output_resolution,
        position_field=position_field,
        position_field_resolution=position_field_resolution,
    )
    expected_output = np.array([
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,1,1,1,1,0],
        [0,0,0,1,1,1,1,0],
        [0,0,0,1,1,1,1,0],
        [0,0,0,1,1,1,1,0],
        [0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0],
    ])

    assert np.allclose(deformed_subject, expected_output)

    # Test constant shifting position_field, demonstrating idiosyncratic extrapolation behavior.

    subject = np.array([
        [0,0,0,0],
        [0,1,1,0],
        [0,1,1,0],
        [0,0,0,0],
    ])
    subject_resolution = 1
    output_resolution = 1
    position_field_resolution = subject_resolution
    position_field = _lddmm_utilities._compute_coords(subject.shape, position_field_resolution) + [0, -1] # Shift to the left by 1.

    deformed_subject = _apply_position_field(
        subject=subject,
        subject_resolution=subject_resolution,
        output_resolution=output_resolution,
        position_field=position_field,
        position_field_resolution=position_field_resolution,
    )
    expected_output = np.array([
        [0,0,0,0],
        [-1,0,1,1],
        [-1,0,1,1],
        [0,0,0,0],
    ])

    assert np.allclose(deformed_subject, expected_output)

    # Test rotational position_field.

    # Note: applying an affine indicating a clockwise-rotation to a position_field produces a position _ield rotated counter-clockwise.
    # The corresponding effect on the deformed_subject is a counter-clockwise rotation.

    subject = np.array([
        [0,1,0,0],
        [0,1,0,0],
        [0,1,0,0],
        [0,1,1,1],
    ])
    subject_resolution = 1
    output_resolution = 1
    position_field_resolution = subject_resolution
    # Indicates a 90 degree rotation to the right.
    affine = np.array([
        [0,1,0],
        [-1,0,0],
        [0,0,1],
    ])
    position_field = _lddmm_utilities._multiply_coords_by_affine(affine, 
        _lddmm_utilities._compute_coords(subject.shape, position_field_resolution))

    deformed_subject = _apply_position_field(
        subject=subject,
        subject_resolution=subject_resolution,
        output_resolution=output_resolution,
        position_field=position_field,
        position_field_resolution=position_field_resolution,
    )
    expected_output = np.array([
        [0,0,0,1],
        [0,0,0,1],
        [1,1,1,1],
        [0,0,0,0],
    ])
    
    assert np.allclose(deformed_subject, expected_output)

"""
Test apply_lddmm.
"""

@pytest.mark.parametrize('deform_to', ['template', 'target'])
def test_apply_lddmm(deform_to):

    # Test identity position fields.
    
    subject = np.array([
        [0,0,0,0],
        [0,1,1,0],
        [0,1,1,0],
        [0,0,0,0],
    ])
    subject_resolution = 1
    output_resolution = None
    template_shape = (3,4)
    template_resolution = 1
    target_shape = (2,5)
    target_resolution = 1
    extrapolation_fill_value = np.quantile(subject, 10**-subject.ndim)

    affine_phi = _lddmm_utilities._compute_coords(template_shape, template_resolution)
    phi_inv_affine_inv = _lddmm_utilities._compute_coords(target_shape, target_resolution)

    expected_output = _apply_position_field(
        subject,
        subject_resolution,
        output_resolution,
        position_field=affine_phi if deform_to == 'template' else phi_inv_affine_inv,
        position_field_resolution=template_resolution if deform_to == 'template' else target_resolution,
        extrapolation_fill_value=extrapolation_fill_value,
    )

    deformed_subject = apply_lddmm(
        subject=subject,                         subject_resolution=subject_resolution, 
        output_resolution=output_resolution,     deform_to=deform_to, 
        extrapolation_fill_value=extrapolation_fill_value,
        affine_phi=affine_phi,                   phi_inv_affine_inv=phi_inv_affine_inv, 
        template_resolution=template_resolution, target_resolution=target_resolution, 
    )

    assert np.allclose(deformed_subject, expected_output)
    assert np.array_equal(deformed_subject, expected_output)

"""
Test lddmm_register.
"""

def _test_lddmm_register(rtol=0, atol=1-1e-9, **lddmm_register_kwargs):

    reg_output = lddmm_register(**lddmm_register_kwargs)

    template            = lddmm_register_kwargs['template']
    target              = lddmm_register_kwargs['target']
    template_resolution = lddmm_register_kwargs['template_resolution'] if 'template_resolution' in lddmm_register_kwargs.keys() else 1
    target_resolution   = lddmm_register_kwargs[  'target_resolution'] if   'target_resolution' in lddmm_register_kwargs.keys() else 1

    deformed_target = apply_lddmm(
        subject=target, 
        subject_resolution=target_resolution, 
        deform_to='template', 
        **reg_output,
    )

    deformed_template = apply_lddmm(
        subject=template, 
        subject_resolution=template_resolution, 
        deform_to='target', 
        **reg_output,
    )

    assert np.allclose(deformed_template, target, rtol=rtol, atol=atol)
    assert np.allclose(deformed_target, template, rtol=rtol, atol=atol)

def test_lddmm_register():

    # Test 2D identity registration.

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
    )

    # _test_lddmm_register(**lddmm_register_kwargs)#debugsuspend

    # Test 3D identity registration.

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
    )

    # _test_lddmm_register(**lddmm_register_kwargs)#debugsuspend

    # Test 4D identity registration.

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
        # affine_stepsize=0.1,
        track_progress_every_n=1,
    )

    _test_lddmm_register(**lddmm_register_kwargs);return
    
    # Test identity quasi-two-dimensional sphere to sphere registration.

    template = np.array([[[(col-6)**2 + (row-6)**2 <= 4**2 for col in range(13)] for row in range(13)]]*2, int)
    target   = np.array([[[(col-6)**2 + (row-6)**2 <= 4**2 for col in range(13)] for row in range(13)]]*2, int)

    lddmm_register_kwargs = dict(
        template=template,
        target=target,
        affine_stepsize=0.01,
        deformative_stepsize=1e-3,
        sigma_regularization=2,
        contrast_order=1,
        track_progress_every_n=1,
    )

    _test_lddmm_register(**lddmm_register_kwargs);return # still working on the above tests.
    
    # Test affine-only quasi-two-dimensional affine-only ellipse to ellipse registration.

    # template has shape (3, 9, 17) and semi-radii 2 and 6.
    template = np.array([[[(col-14)**2/10**2 + (row-10)**2/6**2 <= 1 for col in range(29)] for row in range(21)]]*2, int)
    # target is a rotation of template.
    target = rotate(template, 90, (1,2))
    template_resolution = 1
    target_resolution = 1
    num_iterations = 200
    num_affine_only_iterations = 200
    translational_stepsize = 0.00001
    linear_stepsize = 0.00001
    deformative_stepsize = 0
    sigmaR = 1e10
    smooth_length = None
    num_timesteps = 5
    contrast_order = 1
    sigmaM=None
    initial_affine = np.eye(4)
    initial_velocity_fields = None

    lddmm_register_kwargs = dict(
        template=template, 
        target=target, 
        template_resolution=template_resolution, 
        target_resolution=target_resolution, 
        num_iterations=num_iterations, 
        num_affine_only_iterations=num_affine_only_iterations, 
        translational_stepsize=translational_stepsize, 
        linear_stepsize=linear_stepsize, 
        deformative_stepsize=deformative_stepsize, 
        sigmaR=sigmaR, 
        smooth_length=smooth_length, 
        num_timesteps=num_timesteps, 
        contrast_order=contrast_order, 
        sigmaM=sigmaM, 
        initial_affine=initial_affine, 
        initial_velocity_fields=initial_velocity_fields, 
    )

    _test_lddmm_register(**lddmm_register_kwargs)

    # Test deformative-only quasi-two-dimensional sphere to ellipsoid registration.

    # template has shape (2, 25, 25) and radius 8.
    template = np.array([[[(col-12)**2 + (row-12)**2 <= 8**2 for col in range(25)] for row in range(25)]]*2, int)
    # target has shape (2, 21, 29) and semi-radii 6 and 10.
    target = np.array([[[(col-14)**2/10**2 + (row-10)**2/6**2 <= 1 for col in range(29)] for row in range(21)]]*2, int)
    template_resolution = 1
    target_resolution = 1
    num_iterations = 150
    num_affine_only_iterations = 0
    translational_stepsize = 0
    linear_stepsize = 0
    deformative_stepsize = 0.5
    sigmaR = 1e10
    smooth_length = None
    num_timesteps = 5
    contrast_order = 1
    sigmaM=None
    initial_affine = np.eye(4)
    initial_velocity_fields = None

    lddmm_register_kwargs = dict(
        template=template, 
        target=target, 
        template_resolution=template_resolution, 
        target_resolution=target_resolution, 
        num_iterations=num_iterations, 
        num_affine_only_iterations=num_affine_only_iterations, 
        translational_stepsize=translational_stepsize, 
        linear_stepsize=linear_stepsize, 
        deformative_stepsize=deformative_stepsize, 
        sigmaR=sigmaR, 
        smooth_length=smooth_length, 
        num_timesteps=num_timesteps, 
        contrast_order=contrast_order, 
        sigmaM=sigmaM, 
        initial_affine=initial_affine, 
        initial_velocity_fields=initial_velocity_fields, 
    )

    _test_lddmm_register(**lddmm_register_kwargs)

    # Test deformative and affine quasi-two-dimensional ellipsoid to ellipsoid registration.

    # target has shape (2, 21, 29) and semi-radii 6 and 10.
    template = np.array([[[(col-14)**2/10**2 + (row-10)**2/6**2 <= 1 for col in range(29)] for row in range(21)]]*2, int)
    # target has shape (2, 21, 29) and semi-radii 6 and 10.
    target = rotate(template, 30, (1,2))
    template_resolution = 1
    target_resolution = 1
    num_iterations = 200
    num_affine_only_iterations = 50
    translational_stepsize = 0.00001
    linear_stepsize = 0.00001
    deformative_stepsize = 0.5
    sigmaR = None
    smooth_length = None
    num_timesteps = 5
    contrast_order = 1
    sigmaM=None
    initial_affine = np.eye(4)
    initial_velocity_fields = None

    lddmm_register_kwargs = dict(
        template=template, 
        target=target, 
        template_resolution=template_resolution, 
        target_resolution=target_resolution, 
        num_iterations=num_iterations, 
        num_affine_only_iterations=num_affine_only_iterations, 
        translational_stepsize=translational_stepsize, 
        linear_stepsize=linear_stepsize, 
        deformative_stepsize=deformative_stepsize, 
        sigmaR=sigmaR, 
        smooth_length=smooth_length, 
        num_timesteps=num_timesteps, 
        contrast_order=contrast_order, 
        sigmaM=sigmaM, 
        initial_affine=initial_affine, 
        initial_velocity_fields=initial_velocity_fields, 
    )

    _test_lddmm_register(**lddmm_register_kwargs)

if __name__ == "__main__":
    test__generate_position_field(deform_to='template')
    test__generate_position_field(deform_to='target')
    test__apply_position_field()
    test_apply_lddmm(deform_to='template')
    test_apply_lddmm(deform_to='target')
    test_lddmm_register()
