import math

import numpy as np
from numpy import array

from skimage._shared._warnings import expected_warnings
from skimage.measure._regionprops import (regionprops, PROPS, perimeter,
                                          _parse_docs, _props_to_dict,
                                          regionprops_table, OBJECT_COLUMNS,
                                          COL_DTYPES)
from skimage._shared import testing
from skimage._shared.testing import (assert_array_equal, assert_almost_equal,
                                     assert_array_almost_equal, assert_equal)


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
INTENSITY_SAMPLE = SAMPLE.copy()
INTENSITY_SAMPLE[1, 9:11] = 2

SAMPLE_3D = np.zeros((6, 6, 6), dtype=np.uint8)
SAMPLE_3D[1:3, 1:3, 1:3] = 1
SAMPLE_3D[3, 2, 2] = 1
INTENSITY_SAMPLE_3D = SAMPLE_3D.copy()


def test_all_props():
    region = regionprops(SAMPLE, INTENSITY_SAMPLE)[0]
    for prop in PROPS:
        try:
            assert_almost_equal(region[prop], getattr(region, PROPS[prop]))
        except TypeError:  # the `slice` property causes this
            pass


def test_all_props_3d():
    region = regionprops(SAMPLE_3D, INTENSITY_SAMPLE_3D)[0]
    for prop in PROPS:
        try:
            assert_almost_equal(region[prop], getattr(region, PROPS[prop]))
        except (NotImplementedError, TypeError):
            pass


def test_dtype():
    regionprops(np.zeros((10, 10), dtype=np.int))
    regionprops(np.zeros((10, 10), dtype=np.uint))
    with testing.raises(TypeError):
        regionprops(np.zeros((10, 10), dtype=np.float))
    with testing.raises(TypeError):
        regionprops(np.zeros((10, 10), dtype=np.double))
    with testing.raises(TypeError):
        regionprops(np.zeros((10, 10), dtype=np.bool))


def test_ndim():
    regionprops(np.zeros((10, 10), dtype=np.int))
    regionprops(np.zeros((10, 10, 1), dtype=np.int))
    regionprops(np.zeros((10, 10, 10), dtype=np.int))
    regionprops(np.zeros((1, 1), dtype=np.int))
    regionprops(np.zeros((1, 1, 1), dtype=np.int))
    with testing.raises(TypeError):
        regionprops(np.zeros((10, 10, 10, 2), dtype=np.int))


def test_area():
    area = regionprops(SAMPLE)[0].area
    assert area == np.sum(SAMPLE)
    area = regionprops(SAMPLE_3D)[0].area
    assert area == np.sum(SAMPLE_3D)


def test_bbox():
    bbox = regionprops(SAMPLE)[0].bbox
    assert_array_almost_equal(bbox, (0, 0, SAMPLE.shape[0], SAMPLE.shape[1]))

    SAMPLE_mod = SAMPLE.copy()
    SAMPLE_mod[:, -1] = 0
    bbox = regionprops(SAMPLE_mod)[0].bbox
    assert_array_almost_equal(bbox, (0, 0, SAMPLE.shape[0], SAMPLE.shape[1]-1))

    bbox = regionprops(SAMPLE_3D)[0].bbox
    assert_array_almost_equal(bbox, (1, 1, 1, 4, 3, 3))


def test_bbox_area():
    padded = np.pad(SAMPLE, 5, mode='constant')
    bbox_area = regionprops(padded)[0].bbox_area
    assert_array_almost_equal(bbox_area, SAMPLE.size)


def test_moments_central():
    mu = regionprops(SAMPLE)[0].moments_central
    # determined with OpenCV
    assert_almost_equal(mu[2, 0], 436.00000000000045)
    # different from OpenCV results, bug in OpenCV
    assert_almost_equal(mu[3, 0], -737.333333333333)
    assert_almost_equal(mu[1, 1], -87.33333333333303)
    assert_almost_equal(mu[2, 1], -127.5555555555593)
    assert_almost_equal(mu[0, 2], 1259.7777777777774)
    assert_almost_equal(mu[1, 2], 2000.296296296291)
    assert_almost_equal(mu[0, 3], -760.0246913580195)


def test_centroid():
    centroid = regionprops(SAMPLE)[0].centroid
    # determined with MATLAB
    assert_array_almost_equal(centroid, (5.66666666666666, 9.444444444444444))


def test_centroid_3d():
    centroid = regionprops(SAMPLE_3D)[0].centroid
    # determined by mean along axis 1 of SAMPLE_3D.nonzero()
    assert_array_almost_equal(centroid, (1.66666667, 1.55555556, 1.55555556))


def test_convex_area():
    area = regionprops(SAMPLE)[0].convex_area
    # determined with MATLAB
    assert area == 124


def test_convex_image():
    img = regionprops(SAMPLE)[0].convex_image
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


def test_coordinates():
    sample = np.zeros((10, 10), dtype=np.int8)
    coords = np.array([[3, 2], [3, 3], [3, 4]])
    sample[coords[:, 0], coords[:, 1]] = 1
    prop_coords = regionprops(sample)[0].coords
    assert_array_equal(prop_coords, coords)

    sample = np.zeros((6, 6, 6), dtype=np.int8)
    coords = np.array([[1, 1, 1], [1, 2, 1], [1, 3, 1]])
    sample[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
    prop_coords = regionprops(sample)[0].coords
    assert_array_equal(prop_coords, coords)


def test_slice():
    padded = np.pad(SAMPLE, ((2, 4), (5, 2)), mode='constant')
    nrow, ncol = SAMPLE.shape
    result = regionprops(padded)[0].slice
    expected = (slice(2, 2+nrow), slice(5, 5+ncol))
    assert_equal(result, expected)


def test_eccentricity():
    eps = regionprops(SAMPLE)[0].eccentricity
    assert_almost_equal(eps, 0.814629313427)

    img = np.zeros((5, 5), dtype=np.int)
    img[2, 2] = 1
    eps = regionprops(img)[0].eccentricity
    assert_almost_equal(eps, 0)


def test_equiv_diameter():
    diameter = regionprops(SAMPLE)[0].equivalent_diameter
    # determined with MATLAB
    assert_almost_equal(diameter, 9.57461472963)


def test_euler_number():
    en = regionprops(SAMPLE)[0].euler_number
    assert en == 1

    SAMPLE_mod = SAMPLE.copy()
    SAMPLE_mod[7, -3] = 0
    en = regionprops(SAMPLE_mod)[0].euler_number
    assert en == 0


def test_extent():
    extent = regionprops(SAMPLE)[0].extent
    assert_almost_equal(extent, 0.4)


def test_moments_hu():
    hu = regionprops(SAMPLE)[0].moments_hu
    ref = np.array([
         3.27117627e-01,
         2.63869194e-02,
         2.35390060e-02,
         1.23151193e-03,
         1.38882330e-06,
         -2.72586158e-05,
         -6.48350653e-06
    ])
    # bug in OpenCV caused in Central Moments calculation?
    assert_array_almost_equal(hu, ref)


def test_image():
    img = regionprops(SAMPLE)[0].image
    assert_array_equal(img, SAMPLE)

    img = regionprops(SAMPLE_3D)[0].image
    assert_array_equal(img, SAMPLE_3D[1:4, 1:3, 1:3])


def test_label():
    label = regionprops(SAMPLE)[0].label
    assert_array_equal(label, 1)

    label = regionprops(SAMPLE_3D)[0].label
    assert_array_equal(label, 1)


def test_filled_area():
    area = regionprops(SAMPLE)[0].filled_area
    assert area == np.sum(SAMPLE)

    SAMPLE_mod = SAMPLE.copy()
    SAMPLE_mod[7, -3] = 0
    area = regionprops(SAMPLE_mod)[0].filled_area
    assert area == np.sum(SAMPLE)


def test_filled_image():
    img = regionprops(SAMPLE)[0].filled_image
    assert_array_equal(img, SAMPLE)


def test_major_axis_length():
    length = regionprops(SAMPLE)[0].major_axis_length
    # MATLAB has different interpretation of ellipse than found in literature,
    # here implemented as found in literature
    assert_almost_equal(length, 16.7924234999)


def test_max_intensity():
    intensity = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE
                            )[0].max_intensity
    assert_almost_equal(intensity, 2)


def test_mean_intensity():
    intensity = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE
                            )[0].mean_intensity
    assert_almost_equal(intensity, 1.02777777777777)


def test_min_intensity():
    intensity = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE
                            )[0].min_intensity
    assert_almost_equal(intensity, 1)


def test_minor_axis_length():
    length = regionprops(SAMPLE)[0].minor_axis_length
    # MATLAB has different interpretation of ellipse than found in literature,
    # here implemented as found in literature
    assert_almost_equal(length, 9.739302807263)


def test_moments():
    m = regionprops(SAMPLE)[0].moments
    # determined with OpenCV
    assert_almost_equal(m[0, 0], 72.0)
    assert_almost_equal(m[0, 1], 680.0)
    assert_almost_equal(m[0, 2], 7682.0)
    assert_almost_equal(m[0, 3], 95588.0)
    assert_almost_equal(m[1, 0], 408.0)
    assert_almost_equal(m[1, 1], 3766.0)
    assert_almost_equal(m[1, 2], 43882.0)
    assert_almost_equal(m[2, 0], 2748.0)
    assert_almost_equal(m[2, 1], 24836.0)
    assert_almost_equal(m[3, 0], 19776.0)


def test_moments_normalized():
    nu = regionprops(SAMPLE)[0].moments_normalized

    # determined with OpenCV
    assert_almost_equal(nu[0, 2], 0.24301268861454037)
    assert_almost_equal(nu[0, 3], -0.017278118992041805)
    assert_almost_equal(nu[1, 1], -0.016846707818929982)
    assert_almost_equal(nu[1, 2], 0.045473992910668816)
    assert_almost_equal(nu[2, 0], 0.08410493827160502)
    assert_almost_equal(nu[2, 1], -0.002899800614433943)


def test_orientation():
    orient = regionprops(SAMPLE)[0].orientation
    # determined with MATLAB
    assert_almost_equal(orient, -1.4663278802756865)
    # test diagonal regions
    diag = np.eye(10, dtype=int)
    orient_diag = regionprops(diag)[0].orientation
    assert_almost_equal(orient_diag, -math.pi / 4)
    orient_diag = regionprops(np.flipud(diag))[0].orientation
    assert_almost_equal(orient_diag, math.pi / 4)
    orient_diag = regionprops(np.fliplr(diag))[0].orientation
    assert_almost_equal(orient_diag, math.pi / 4)
    orient_diag = regionprops(np.fliplr(np.flipud(diag)))[0].orientation
    assert_almost_equal(orient_diag, -math.pi / 4)


def test_perimeter():
    per = regionprops(SAMPLE)[0].perimeter
    assert_almost_equal(per, 55.2487373415)

    per = perimeter(SAMPLE.astype('double'), neighbourhood=8)
    assert_almost_equal(per, 46.8284271247)


def test_solidity():
    solidity = regionprops(SAMPLE)[0].solidity
    # determined with MATLAB
    assert_almost_equal(solidity, 0.580645161290323)


def test_weighted_moments_central():
    wmu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE
                      )[0].weighted_moments_central
    ref = np.array(
        [[7.4000000000e+01, 3.7303493627e-14, 1.2602837838e+03,
          -7.6561796932e+02],
         [-2.1316282073e-13, -8.7837837838e+01, 2.1571526662e+03,
          -4.2385971907e+03],
         [4.7837837838e+02, -1.4801314828e+02, 6.6989799420e+03,
          -9.9501164076e+03],
         [-7.5943608473e+02, -1.2714707125e+03, 1.5304076361e+04,
          -3.3156729271e+04]])

    np.set_printoptions(precision=10)
    assert_array_almost_equal(wmu, ref)


def test_weighted_centroid():
    centroid = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE
                           )[0].weighted_centroid
    assert_array_almost_equal(centroid, (5.540540540540, 9.445945945945))


def test_weighted_moments_hu():
    whu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE
                      )[0].weighted_moments_hu
    ref = np.array([
        3.1750587329e-01,
        2.1417517159e-02,
        2.3609322038e-02,
        1.2565683360e-03,
        8.3014209421e-07,
        -3.5073773473e-05,
        -6.7936409056e-06
    ])
    assert_array_almost_equal(whu, ref)


def test_weighted_moments():
    wm = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE
                     )[0].weighted_moments
    ref = np.array(
        [[7.4000000e+01, 6.9900000e+02, 7.8630000e+03, 9.7317000e+04],
         [4.1000000e+02, 3.7850000e+03, 4.4063000e+04, 5.7256700e+05],
         [2.7500000e+03, 2.4855000e+04, 2.9347700e+05, 3.9007170e+06],
         [1.9778000e+04, 1.7500100e+05, 2.0810510e+06, 2.8078871e+07]]
    )
    assert_array_almost_equal(wm, ref)


def test_weighted_moments_normalized():
    wnu = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE
                      )[0].weighted_moments_normalized
    ref = np.array(
        [[       np.nan,        np.nan, 0.2301467830, -0.0162529732],
         [       np.nan, -0.0160405109, 0.0457932622, -0.0104598869],
         [ 0.0873590903, -0.0031421072, 0.0165315478, -0.0028544152],
         [-0.0161217406, -0.0031376984, 0.0043903193, -0.0011057191]]
    )
    assert_array_almost_equal(wnu, ref)


def test_label_sequence():
    a = np.empty((2, 2), dtype=np.int)
    a[:, :] = 2
    ps = regionprops(a)
    assert len(ps) == 1
    assert ps[0].label == 2


def test_pure_background():
    a = np.zeros((2, 2), dtype=np.int)
    ps = regionprops(a)
    assert len(ps) == 0


def test_invalid():
    ps = regionprops(SAMPLE)

    def get_intensity_image():
        ps[0].intensity_image

    with testing.raises(AttributeError):
        get_intensity_image()


def test_invalid_size():
    wrong_intensity_sample = np.array([[1], [1]])
    with testing.raises(ValueError):
        regionprops(SAMPLE, wrong_intensity_sample)


def test_equals():
    arr = np.zeros((100, 100), dtype=np.int)
    arr[0:25, 0:25] = 1
    arr[50:99, 50:99] = 2

    regions = regionprops(arr)
    r1 = regions[0]

    regions = regionprops(arr)
    r2 = regions[0]
    r3 = regions[1]

    assert_equal(r1 == r2, True, "Same regionprops are not equal")
    assert_equal(r1 != r3, True, "Different regionprops are equal")


def test_iterate_all_props():
    region = regionprops(SAMPLE)[0]
    p0 = {p: region[p] for p in region}

    region = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0]
    p1 = {p: region[p] for p in region}

    assert len(p0) < len(p1)


def test_cache():
    SAMPLE_mod = SAMPLE.copy()
    region = regionprops(SAMPLE_mod)[0]
    f0 = region.filled_image
    region._label_image[:10] = 1
    f1 = region.filled_image

    # Changed underlying image, but cache keeps result the same
    assert_array_equal(f0, f1)

    # Now invalidate cache
    region._cache_active = False
    f1 = region.filled_image

    assert np.any(f0 != f1)


def test_docstrings_and_props():
    def foo():
        """foo"""

    has_docstrings = bool(foo.__doc__)

    region = regionprops(SAMPLE)[0]

    docs = _parse_docs()
    props = [m for m in dir(region) if not m.startswith('_')]

    nr_docs_parsed = len(docs)
    nr_props = len(props)
    if has_docstrings:
        assert_equal(nr_docs_parsed, nr_props)
        ds = docs['weighted_moments_normalized']
        assert 'iteration' not in ds
        assert len(ds.split('\n')) > 3
    else:
        assert_equal(nr_docs_parsed, 0)


def test_props_to_dict():
    regions = regionprops(SAMPLE)
    out = _props_to_dict(regions)
    assert out == {'label': array([1]),
                   'bbox-0': array([0]), 'bbox-1': array([0]),
                   'bbox-2': array([10]), 'bbox-3': array([18])}

    regions = regionprops(SAMPLE)
    out = _props_to_dict(regions, properties=('label', 'area', 'bbox'),
                         separator='+')
    assert out == {'label': array([1]), 'area': array([72]),
                   'bbox+0': array([0]), 'bbox+1': array([0]),
                   'bbox+2': array([10]), 'bbox+3': array([18])}


def test_regionprops_table():
    out = regionprops_table(SAMPLE)
    assert out == {'label': array([1]),
                   'bbox-0': array([0]), 'bbox-1': array([0]),
                   'bbox-2': array([10]), 'bbox-3': array([18])}

    out = regionprops_table(SAMPLE, properties=('label', 'area', 'bbox'),
                            separator='+')
    assert out == {'label': array([1]), 'area': array([72]),
                   'bbox+0': array([0]), 'bbox+1': array([0]),
                   'bbox+2': array([10]), 'bbox+3': array([18])}

    out = regionprops_table(np.zeros((2, 2), dtype=int),
                            properties=('label', 'area', 'bbox'),
                            separator='+')
    assert len(out) == 6
    assert len(out['label']) == 0
    assert len(out['area']) == 0
    assert len(out['bbox+0']) == 0
    assert len(out['bbox+1']) == 0
    assert len(out['bbox+2']) == 0
    assert len(out['bbox+3']) == 0


def test_props_dict_complete():
    region = regionprops(SAMPLE)[0]
    properties = [s for s in dir(region) if not s.startswith('_')]
    assert set(properties) == set(PROPS.values())


def test_column_dtypes_complete():
    assert set(COL_DTYPES.keys()).union(OBJECT_COLUMNS) == set(PROPS.values())


def test_column_dtypes_correct():
    msg = 'mismatch with expected type,'
    region = regionprops(SAMPLE, intensity_image=INTENSITY_SAMPLE)[0]
    for col in COL_DTYPES:
        r = region[col]

        if col in OBJECT_COLUMNS:
            assert COL_DTYPES[col] == object
            continue

        t = type(np.ravel(r)[0])

        if np.issubdtype(t, np.floating):
            assert COL_DTYPES[col] == float, (
                f'{col} dtype {t} {msg} {COL_DTYPES[col]}'
            )
        elif np.issubdtype(t, np.integer):
            assert COL_DTYPES[col] == int, (
                f'{col} dtype {t} {msg} {COL_DTYPES[col]}'
            )
        else:
            assert False, (
                f'{col} dtype {t} {msg} {COL_DTYPES[col]}'
            )


def test_deprecated_coords_argument():
    with expected_warnings(['coordinates keyword argument']):
        region = regionprops(SAMPLE, coordinates='rc')
    with testing.raises(ValueError):
        region = regionprops(SAMPLE, coordinates='xy')
