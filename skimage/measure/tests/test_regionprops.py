from numpy.testing import assert_array_equal, assert_almost_equal, \
    assert_array_almost_equal
import numpy as np

from skimage.measure import regionprops


SAMPLE = np.array(
    [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
     [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0],
     [0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1],
     [0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]
)


def test_area():
    area = regionprops(SAMPLE, ['Area'])[0]['Area']
    assert area == np.sum(SAMPLE)

def test_bbox():
    bbox = regionprops(SAMPLE, ['BoundingBox'])[0]['BoundingBox']
    assert_array_almost_equal(bbox, (0, 0, SAMPLE.shape[0], SAMPLE.shape[1]))

    SAMPLE_mod = SAMPLE.copy()
    SAMPLE_mod[:,-1] = 0
    bbox = regionprops(SAMPLE_mod, ['BoundingBox'])[0]['BoundingBox']
    assert_array_almost_equal(bbox, (0, 0, SAMPLE.shape[0], SAMPLE.shape[1]-1))

def test_central_moments():
    mu = regionprops(SAMPLE, ['CentralMoments'])[0]['CentralMoments']
    #: determined with OpenCV
    assert_almost_equal(mu[0,2], 436.00000000000045)
    # different from OpenCV results, bug in OpenCV
    assert_almost_equal(mu[0,3], -737.333333333333)
    assert_almost_equal(mu[1,1], -87.33333333333303)
    assert_almost_equal(mu[1,2], -127.5555555555593)
    assert_almost_equal(mu[2,0], 1259.7777777777774)
    assert_almost_equal(mu[2,1], 2000.296296296291)
    assert_almost_equal(mu[3,0], -760.0246913580195)

def test_centroid():
    centroid = regionprops(SAMPLE, ['Centroid'])[0]['Centroid']
    # determined with MATLAB
    assert_array_almost_equal(centroid, (5.66666666666666, 9.444444444444444))

def test_convex_area():
    area = regionprops(SAMPLE, ['ConvexArea'])[0]['ConvexArea']
    # determined with MATLAB
    assert area == 124

def test_convex_image():
    img = regionprops(SAMPLE, ['ConvexImage'])[0]['ConvexImage']
    # determined with MATLAB
    ref = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
         [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    )
    assert_array_equal(img, ref)

def test_eccentricity():
    eps = regionprops(SAMPLE, ['Eccentricity'])[0]['Eccentricity']
    assert_almost_equal(eps, 0.814629313427)

def test_equiv_diameter():
    diameter = regionprops(SAMPLE, ['EquivDiameter'])[0]['EquivDiameter']
    # determined with MATLAB
    assert_almost_equal(diameter, 9.57461472963)

def test_euler_number():
    en = regionprops(SAMPLE, ['EulerNumber'])[0]['EulerNumber']
    assert en == 0

    SAMPLE_mod = SAMPLE.copy()
    SAMPLE_mod[7,-3] = 0
    en = regionprops(SAMPLE_mod, ['EulerNumber'])[0]['EulerNumber']
    assert en == -1

def test_extent():
    extent = regionprops(SAMPLE, ['Extent'])[0]['Extent']
    assert_almost_equal(extent, 0.4)

def test_hu_moments():
    hu = regionprops(SAMPLE, ['HuMoments'])[0]['HuMoments']
    ref = np.array([
        3.27117627e-01,
        2.63869194e-02,
        2.35390060e-02,
        1.23151193e-03,
        1.38882330e-06,
        -2.72586158e-05,
        6.48350653e-06
    ])
    # bug in OpenCV caused in Central Moments calculation?
    assert_array_almost_equal(hu, ref)

def test_image():
    img = regionprops(SAMPLE, ['Image'])[0]['Image']
    assert_array_equal(img, SAMPLE)

def test_filled_area():
    area = regionprops(SAMPLE, ['FilledArea'])[0]['FilledArea']
    assert area == np.sum(SAMPLE)

    SAMPLE_mod = SAMPLE.copy()
    SAMPLE_mod[7,-3] = 0
    area = regionprops(SAMPLE_mod, ['FilledArea'])[0]['FilledArea']
    assert area == np.sum(SAMPLE)

def test_minor_axis_length():
    length = regionprops(SAMPLE, ['MinorAxisLength'])[0]['MinorAxisLength']
    # MATLAB has different interpretation of ellipse than found in literature,
    # here implemented as found in literature
    assert_almost_equal(length, 9.739302807263)

def test_major_axis_length():
    length = regionprops(SAMPLE, ['MajorAxisLength'])[0]['MajorAxisLength']
    # MATLAB has different interpretation of ellipse than found in literature,
    # here implemented as found in literature
    assert_almost_equal(length, 16.7924234999)

def test_moments():
    m = regionprops(SAMPLE, ['Moments'])[0]['Moments']
    #: determined with OpenCV
    assert_almost_equal(m[0,0], 72.0)
    assert_almost_equal(m[0,1], 408.0)
    assert_almost_equal(m[0,2], 2748.0)
    assert_almost_equal(m[0,3], 19776.0)
    assert_almost_equal(m[1,0], 680.0)
    assert_almost_equal(m[1,1], 3766.0)
    assert_almost_equal(m[1,2], 24836.0)
    assert_almost_equal(m[2,0], 7682.0)
    assert_almost_equal(m[2,1], 43882.0)
    assert_almost_equal(m[3,0], 95588.0)

def test_normalized_moments():
    nu = regionprops(SAMPLE, ['NormalizedMoments'])[0]['NormalizedMoments']
    #: determined with OpenCV
    assert_almost_equal(nu[0,2], 0.08410493827160502)
    assert_almost_equal(nu[1,1], -0.016846707818929982)
    assert_almost_equal(nu[1,2], -0.002899800614433943)
    assert_almost_equal(nu[2,0], 0.24301268861454037)
    assert_almost_equal(nu[2,1], 0.045473992910668816)
    assert_almost_equal(nu[3,0], -0.017278118992041805)

def test_orientation():
    orientation = regionprops(SAMPLE, ['Orientation'])[0]['Orientation']
    # determined with MATLAB
    assert_almost_equal(orientation, 0.10446844651921)

def test_solidity():
    solidity = regionprops(SAMPLE, ['Solidity'])[0]['Solidity']
    # determined with MATLAB
    assert_almost_equal(solidity, 0.580645161290323)


if __name__ == "__main__":
    from numpy.testing import run_module_suite
    run_module_suite()
