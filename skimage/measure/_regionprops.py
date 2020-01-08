from warnings import warn
from math import sqrt, atan2, pi as PI
import numpy as np
from scipy import ndimage as ndi

from ._label import label
from . import _moments

from functools import wraps


__all__ = ['regionprops', 'perimeter']


STREL_4 = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]], dtype=np.uint8)
STREL_8 = np.ones((3, 3), dtype=np.uint8)
STREL_26_3D = np.ones((3, 3, 3), dtype=np.uint8)
PROPS = {
    'Area': 'area',
    'BoundingBox': 'bbox',
    'BoundingBoxArea': 'bbox_area',
    'CentralMoments': 'moments_central',
    'Centroid': 'centroid',
    'ConvexArea': 'convex_area',
    # 'ConvexHull',
    'ConvexImage': 'convex_image',
    'Coordinates': 'coords',
    'Eccentricity': 'eccentricity',
    'EquivDiameter': 'equivalent_diameter',
    'EulerNumber': 'euler_number',
    'Extent': 'extent',
    # 'Extrema',
    'FilledArea': 'filled_area',
    'FilledImage': 'filled_image',
    'HuMoments': 'moments_hu',
    'Image': 'image',
    'InertiaTensor': 'inertia_tensor',
    'InertiaTensorEigvals': 'inertia_tensor_eigvals',
    'IntensityImage': 'intensity_image',
    'Label': 'label',
    'LocalCentroid': 'local_centroid',
    'MajorAxisLength': 'major_axis_length',
    'MaxIntensity': 'max_intensity',
    'MeanIntensity': 'mean_intensity',
    'MinIntensity': 'min_intensity',
    'MinorAxisLength': 'minor_axis_length',
    'Moments': 'moments',
    'NormalizedMoments': 'moments_normalized',
    'Orientation': 'orientation',
    'Perimeter': 'perimeter',
    # 'PixelIdxList',
    # 'PixelList',
    'Slice': 'slice',
    'Solidity': 'solidity',
    # 'SubarrayIdx'
    'WeightedCentralMoments': 'weighted_moments_central',
    'WeightedCentroid': 'weighted_centroid',
    'WeightedHuMoments': 'weighted_moments_hu',
    'WeightedLocalCentroid': 'weighted_local_centroid',
    'WeightedMoments': 'weighted_moments',
    'WeightedNormalizedMoments': 'weighted_moments_normalized'
}

OBJECT_COLUMNS = {
    'image', 'coords', 'convex_image', 'slice',
    'filled_image', 'intensity_image'
}

COL_DTYPES = {
    'area': int,
    'bbox': int,
    'bbox_area': int,
    'moments_central': float,
    'centroid': float,
    'convex_area': int,
    'convex_image': object,
    'coords': object,
    'eccentricity': float,
    'equivalent_diameter': float,
    'euler_number': int,
    'extent': float,
    'filled_area': int,
    'filled_image': object,
    'moments_hu': float,
    'image': object,
    'inertia_tensor': float,
    'inertia_tensor_eigvals': float,
    'intensity_image': object,
    'label': int,
    'local_centroid': float,
    'major_axis_length': float,
    'max_intensity': int,
    'mean_intensity': float,
    'min_intensity': int,
    'minor_axis_length': float,
    'moments': float,
    'moments_normalized': float,
    'orientation': float,
    'perimeter': float,
    'slice': object,
    'solidity': float,
    'weighted_moments_central': float,
    'weighted_centroid': float,
    'weighted_moments_hu': float,
    'weighted_local_centroid': float,
    'weighted_moments': float,
    'weighted_moments_normalized': float
}

PROP_VALS = set(PROPS.values())


def _cached(f):
    @wraps(f)
    def wrapper(obj):
        cache = obj._cache
        prop = f.__name__

        if not ((prop in cache) and obj._cache_active):
            cache[prop] = f(obj)

        return cache[prop]

    return wrapper


def only2d(method):
    @wraps(method)
    def func2d(self, *args, **kwargs):
        if self._ndim > 2:
            raise NotImplementedError('Property %s is not implemented for '
                                      '3D images' % method.__name__)
        return method(self, *args, **kwargs)
    return func2d


class RegionProperties:
    """Please refer to `skimage.measure.regionprops` for more information
    on the available region properties.
    """

    def __init__(self, slice, label, label_image, intensity_image,
                 cache_active):

        if intensity_image is not None:
            if not intensity_image.shape == label_image.shape:
                raise ValueError('Label and intensity image must have the'
                                 ' same shape.')

        self.label = label

        self._slice = slice
        self.slice = slice
        self._label_image = label_image
        self._intensity_image = intensity_image

        self._cache_active = cache_active
        self._cache = {}
        self._ndim = label_image.ndim

    @property
    @_cached
    def area(self):
        return np.sum(self.image)

    @property
    def bbox(self):
        """
        Returns
        -------
        A tuple of the bounding box's start coordinates for each dimension,
        followed by the end coordinates for each dimension
        """
        return tuple([self.slice[i].start for i in range(self._ndim)] +
                     [self.slice[i].stop for i in range(self._ndim)])

    @property
    def bbox_area(self):
        return self.image.size

    @property
    def centroid(self):
        return tuple(self.coords.mean(axis=0))

    @property
    @_cached
    def convex_area(self):
        return np.sum(self.convex_image)

    @property
    @_cached
    def convex_image(self):
        from ..morphology.convex_hull import convex_hull_image
        return convex_hull_image(self.image)

    @property
    def coords(self):
        indices = np.nonzero(self.image)
        return np.vstack([indices[i] + self.slice[i].start
                          for i in range(self._ndim)]).T

    @property
    @only2d
    def eccentricity(self):
        l1, l2 = self.inertia_tensor_eigvals
        if l1 == 0:
            return 0
        return sqrt(1 - l2 / l1)

    @property
    def equivalent_diameter(self):
        if self._ndim == 2:
            return sqrt(4 * self.area / PI)
        elif self._ndim == 3:
            return (6 * self.area / PI) ** (1. / 3)

    @property
    def euler_number(self):
        euler_array = self.filled_image != self.image
        _, num = label(euler_array, connectivity=self._ndim, return_num=True,
                       background=0)
        return -num + 1

    @property
    def extent(self):
        return self.area / self.image.size

    @property
    def filled_area(self):
        return np.sum(self.filled_image)

    @property
    @_cached
    def filled_image(self):
        structure = np.ones((3,) * self._ndim)
        return ndi.binary_fill_holes(self.image, structure)

    @property
    @_cached
    def image(self):
        return self._label_image[self.slice] == self.label

    @property
    @_cached
    def inertia_tensor(self):
        mu = self.moments_central
        return _moments.inertia_tensor(self.image, mu)

    @property
    @_cached
    def inertia_tensor_eigvals(self):
        return _moments.inertia_tensor_eigvals(self.image,
                                               T=self.inertia_tensor)

    @property
    @_cached
    def intensity_image(self):
        if self._intensity_image is None:
            raise AttributeError('No intensity image specified.')
        return self._intensity_image[self.slice] * self.image

    def _intensity_image_double(self):
        return self.intensity_image.astype(np.double)

    @property
    def local_centroid(self):
        M = self.moments
        return tuple(M[tuple(np.eye(self._ndim, dtype=int))] /
                     M[(0,) * self._ndim])

    @property
    def max_intensity(self):
        return np.max(self.intensity_image[self.image])

    @property
    def mean_intensity(self):
        return np.mean(self.intensity_image[self.image])

    @property
    def min_intensity(self):
        return np.min(self.intensity_image[self.image])

    @property
    def major_axis_length(self):
        l1 = self.inertia_tensor_eigvals[0]
        return 4 * sqrt(l1)

    @property
    def minor_axis_length(self):
        l2 = self.inertia_tensor_eigvals[-1]
        return 4 * sqrt(l2)

    @property
    @_cached
    def moments(self):
        M = _moments.moments(self.image.astype(np.uint8), 3)
        return M

    @property
    @_cached
    def moments_central(self):
        mu = _moments.moments_central(self.image.astype(np.uint8),
                                      self.local_centroid, order=3)
        return mu

    @property
    @only2d
    def moments_hu(self):
        return _moments.moments_hu(self.moments_normalized)

    @property
    @_cached
    def moments_normalized(self):
        return _moments.moments_normalized(self.moments_central, 3)

    @property
    @only2d
    def orientation(self):
        a, b, b, c = self.inertia_tensor.flat
        if a - c == 0:
            if b < 0:
                return -PI / 4.
            else:
                return PI / 4.
        else:
            return 0.5 * atan2(-2 * b, c - a)

    @property
    @only2d
    def perimeter(self):
        return perimeter(self.image, 4)

    @property
    def solidity(self):
        return self.area / self.convex_area

    @property
    def weighted_centroid(self):
        ctr = self.weighted_local_centroid
        return tuple(idx + slc.start
                     for idx, slc in zip(ctr, self.slice))

    @property
    def weighted_local_centroid(self):
        M = self.weighted_moments
        return (M[tuple(np.eye(self._ndim, dtype=int))] /
                M[(0,) * self._ndim])

    @property
    @_cached
    def weighted_moments(self):
        return _moments.moments(self._intensity_image_double(), 3)

    @property
    @_cached
    def weighted_moments_central(self):
        ctr = self.weighted_local_centroid
        return _moments.moments_central(self._intensity_image_double(),
                                        center=ctr, order=3)

    @property
    @only2d
    def weighted_moments_hu(self):
        return _moments.moments_hu(self.weighted_moments_normalized)

    @property
    @_cached
    def weighted_moments_normalized(self):
        return _moments.moments_normalized(self.weighted_moments_central, 3)

    def __iter__(self):
        props = PROP_VALS

        if self._intensity_image is None:
            unavailable_props = ('intensity_image',
                                 'max_intensity',
                                 'mean_intensity',
                                 'min_intensity',
                                 'weighted_moments',
                                 'weighted_moments_central',
                                 'weighted_centroid',
                                 'weighted_local_centroid',
                                 'weighted_moments_hu',
                                 'weighted_moments_normalized')

            props = props.difference(unavailable_props)

        return iter(sorted(props))

    def __getitem__(self, key):
        value = getattr(self, key, None)
        if value is not None:
            return value
        else:  # backwards compatibility
            return getattr(self, PROPS[key])

    def __eq__(self, other):
        if not isinstance(other, RegionProperties):
            return False

        for key in PROP_VALS:
            try:
                # so that NaNs are equal
                np.testing.assert_equal(getattr(self, key, None),
                                        getattr(other, key, None))
            except AssertionError:
                return False

        return True


# For compatibility with code written prior to 0.16
_RegionProperties = RegionProperties


def _props_to_dict(regions, properties=('label', 'bbox'), separator='-'):
    """Convert image region properties list into a column dictionary.

    Parameters
    ----------
    regions : (N,) list
        List of RegionProperties objects as returned by :func:`regionprops`.
    properties : tuple or list of str, optional
        Properties that will be included in the resulting dictionary
        For a list of available properties, please see :func:`regionprops`.
        Users should remember to add "label" to keep track of region
        identities.
    separator : str, optional
        For non-scalar properties not listed in OBJECT_COLUMNS, each element
        will appear in its own column, with the index of that element separated
        from the property name by this separator. For example, the inertia
        tensor of a 2D region will appear in four columns:
        ``inertia_tensor-0-0``, ``inertia_tensor-0-1``, ``inertia_tensor-1-0``,
        and ``inertia_tensor-1-1`` (where the separator is ``-``).

        Object columns are those that cannot be split in this way because the
        number of columns would change depending on the object. For example,
        ``image`` and ``coords``.

    Returns
    -------
    out_dict : dict
        Dictionary mapping property names to an array of values of that
        property, one value per region. This dictionary can be used as input to
        pandas ``DataFrame`` to map property names to columns in the frame and
        regions to rows.

    Notes
    -----
    Each column contains either a scalar property, an object property, or an
    element in a multidimensional array.

    Properties with scalar values for each region, such as "eccentricity", will
    appear as a float or int array with that property name as key.

    Multidimensional properties *of fixed size* for a given image dimension,
    such as "centroid" (every centroid will have three elements in a 3D image,
    no matter the region size), will be split into that many columns, with the
    name {property_name}{separator}{element_num} (for 1D properties),
    {property_name}{separator}{elem_num0}{separator}{elem_num1} (for 2D
    properties), and so on.

    For multidimensional properties that don't have a fixed size, such as
    "image" (the image of a region varies in size depending on the region
    size), an object array will be used, with the corresponding property name
    as the key.

    Examples
    --------
    >>> from skimage import data, util, measure
    >>> image = data.coins()
    >>> label_image = measure.label(image > 110, connectivity=image.ndim)
    >>> proplist = regionprops(label_image, image)
    >>> props = _props_to_dict(proplist, properties=['label', 'inertia_tensor',
    ...                                              'inertia_tensor_eigvals'])
    >>> props  # doctest: +ELLIPSIS +SKIP
    {'label': array([ 1,  2, ...]), ...
     'inertia_tensor-0-0': array([  4.012...e+03,   8.51..., ...]), ...
     ...,
     'inertia_tensor_eigvals-1': array([  2.67...e+02,   2.83..., ...])}

    The resulting dictionary can be directly passed to pandas, if installed, to
    obtain a clean DataFrame:

    >>> import pandas as pd  # doctest: +SKIP
    >>> data = pd.DataFrame(props)  # doctest: +SKIP
    >>> data.head()  # doctest: +SKIP
       label  inertia_tensor-0-0  ...  inertia_tensor_eigvals-1
    0      1         4012.909888  ...                267.065503
    1      2            8.514739  ...                  2.834806
    2      3            0.666667  ...                  0.000000
    3      4            0.000000  ...                  0.000000
    4      5            0.222222  ...                  0.111111

    """

    out = {}
    n = len(regions)
    for prop in properties:
        dtype = COL_DTYPES[prop]
        column_buffer = np.zeros(n, dtype=dtype)
        r = regions[0][prop]

        # scalars and objects are dedicated one column per prop
        # array properties are raveled into multiple columns
        # for more info, refer to notes 1
        if np.isscalar(r) or prop in OBJECT_COLUMNS:
            for i in range(n):
                column_buffer[i] = regions[i][prop]
            out[prop] = np.copy(column_buffer)
        else:
            if isinstance(r, np.ndarray):
                shape = r.shape
            else:
                shape = (len(r),)

            for ind in np.ndindex(shape):
                for k in range(n):
                    loc = ind if len(ind) > 1 else ind[0]
                    column_buffer[k] = regions[k][prop][loc]
                modified_prop = separator.join(map(str, (prop,) + ind))
                out[modified_prop] = np.copy(column_buffer)
    return out


def regionprops_table(label_image, intensity_image=None,
                      properties=('label', 'bbox'),
                      *,
                      cache=True, separator='-'):
    """Compute image properties and return them as a pandas-compatible table.

    The table is a dictionary mapping column names to value arrays. See Notes
    section below for details.

    Parameters
    ----------
    label_image : (N, M) ndarray
        Labeled input image. Labels with value 0 are ignored.
    intensity_image : (N, M) ndarray, optional
        Intensity (i.e., input) image with same size as labeled image.
        Default is None.
    properties : tuple or list of str, optional
        Properties that will be included in the resulting dictionary
        For a list of available properties, please see :func:`regionprops`.
        Users should remember to add "label" to keep track of region
        identities.
    cache : bool, optional
        Determine whether to cache calculated properties. The computation is
        much faster for cached properties, whereas the memory consumption
        increases.
    separator : str, optional
        For non-scalar properties not listed in OBJECT_COLUMNS, each element
        will appear in its own column, with the index of that element separated
        from the property name by this separator. For example, the inertia
        tensor of a 2D region will appear in four columns:
        ``inertia_tensor-0-0``, ``inertia_tensor-0-1``, ``inertia_tensor-1-0``,
        and ``inertia_tensor-1-1`` (where the separator is ``-``).

        Object columns are those that cannot be split in this way because the
        number of columns would change depending on the object. For example,
        ``image`` and ``coords``.

    Returns
    -------
    out_dict : dict
        Dictionary mapping property names to an array of values of that
        property, one value per region. This dictionary can be used as input to
        pandas ``DataFrame`` to map property names to columns in the frame and
        regions to rows. If the image has no regions,
        the arrays will have length 0, but the correct type.

    Notes
    -----
    Each column contains either a scalar property, an object property, or an
    element in a multidimensional array.

    Properties with scalar values for each region, such as "eccentricity", will
    appear as a float or int array with that property name as key.

    Multidimensional properties *of fixed size* for a given image dimension,
    such as "centroid" (every centroid will have three elements in a 3D image,
    no matter the region size), will be split into that many columns, with the
    name {property_name}{separator}{element_num} (for 1D properties),
    {property_name}{separator}{elem_num0}{separator}{elem_num1} (for 2D
    properties), and so on.

    For multidimensional properties that don't have a fixed size, such as
    "image" (the image of a region varies in size depending on the region
    size), an object array will be used, with the corresponding property name
    as the key.

    Examples
    --------
    >>> from skimage import data, util, measure
    >>> image = data.coins()
    >>> label_image = measure.label(image > 110, connectivity=image.ndim)
    >>> props = regionprops_table(label_image, image,
    ...                           properties=['label', 'inertia_tensor',
    ...                                       'inertia_tensor_eigvals'])
    >>> props  # doctest: +ELLIPSIS +SKIP
    {'label': array([ 1,  2, ...]), ...
     'inertia_tensor-0-0': array([  4.012...e+03,   8.51..., ...]), ...
     ...,
     'inertia_tensor_eigvals-1': array([  2.67...e+02,   2.83..., ...])}

    The resulting dictionary can be directly passed to pandas, if installed, to
    obtain a clean DataFrame:

    >>> import pandas as pd  # doctest: +SKIP
    >>> data = pd.DataFrame(props)  # doctest: +SKIP
    >>> data.head()  # doctest: +SKIP
       label  inertia_tensor-0-0  ...  inertia_tensor_eigvals-1
    0      1         4012.909888  ...                267.065503
    1      2            8.514739  ...                  2.834806
    2      3            0.666667  ...                  0.000000
    3      4            0.000000  ...                  0.000000
    4      5            0.222222  ...                  0.111111

    [5 rows x 7 columns]

    """
    regions = regionprops(label_image, intensity_image=intensity_image,
                          cache=cache)

    if len(regions) == 0:
        label_image = np.zeros((3,) * label_image.ndim, dtype=int)
        label_image[(1,) * label_image.ndim] = 1
        if intensity_image is not None:
            intensity_image = np.zeros(label_image.shape,
                                       dtype=intensity_image.dtype)
        regions = regionprops(label_image, intensity_image=intensity_image,
                              cache=cache)

        out_d = _props_to_dict(regions, properties=properties,
                               separator=separator)
        return {k: v[:0] for k, v in out_d.items()}

    return _props_to_dict(regions, properties=properties, separator=separator)


def regionprops(label_image, intensity_image=None, cache=True,
                coordinates=None):
    r"""Measure properties of labeled image regions.

    Parameters
    ----------
    label_image : (N, M) ndarray
        Labeled input image. Labels with value 0 are ignored.

        .. versionchanged:: 0.14.1
            Previously, ``label_image`` was processed by ``numpy.squeeze`` and
            so any number of singleton dimensions was allowed. This resulted in
            inconsistent handling of images with singleton dimensions. To
            recover the old behaviour, use
            ``regionprops(np.squeeze(label_image), ...)``.
    intensity_image : (N, M) ndarray, optional
        Intensity (i.e., input) image with same size as labeled image.
        Default is None.
    cache : bool, optional
        Determine whether to cache calculated properties. The computation is
        much faster for cached properties, whereas the memory consumption
        increases.
    coordinates : DEPRECATED
        This argument is deprecated and will be removed in a future version
        of scikit-image.

        See :ref:`Coordinate conventions <numpy-images-coordinate-conventions>`
        for more details.

        .. deprecated:: 0.16.0
            Use "rc" coordinates everywhere. It may be sufficient to call
            ``numpy.transpose`` on your label image to get the same values as
            0.15 and earlier. However, for some properties, the transformation
            will be less trivial. For example, the new orientation is
            :math:`\frac{\pi}{2}` plus the old orientation.


    Returns
    -------
    properties : list of RegionProperties
        Each item describes one labeled region, and can be accessed using the
        attributes listed below.

    Notes
    -----
    The following properties can be accessed as attributes or keys:

    **area** : int
        Number of pixels of the region.
    **bbox** : tuple
        Bounding box ``(min_row, min_col, max_row, max_col)``.
        Pixels belonging to the bounding box are in the half-open interval
        ``[min_row; max_row)`` and ``[min_col; max_col)``.
    **bbox_area** : int
        Number of pixels of bounding box.
    **centroid** : array
        Centroid coordinate tuple ``(row, col)``.
    **convex_area** : int
        Number of pixels of convex hull image, which is the smallest convex
        polygon that encloses the region.
    **convex_image** : (H, J) ndarray
        Binary convex hull image which has the same size as bounding box.
    **coords** : (N, 2) ndarray
        Coordinate list ``(row, col)`` of the region.
    **eccentricity** : float
        Eccentricity of the ellipse that has the same second-moments as the
        region. The eccentricity is the ratio of the focal distance
        (distance between focal points) over the major axis length.
        The value is in the interval [0, 1).
        When it is 0, the ellipse becomes a circle.
    **equivalent_diameter** : float
        The diameter of a circle with the same area as the region.
    **euler_number** : int
        Euler characteristic of region. Computed as number of objects (= 1)
        subtracted by number of holes (8-connectivity).
    **extent** : float
        Ratio of pixels in the region to pixels in the total bounding box.
        Computed as ``area / (rows * cols)``
    **filled_area** : int
        Number of pixels of the region will all the holes filled in. Describes
        the area of the filled_image.
    **filled_image** : (H, J) ndarray
        Binary region image with filled holes which has the same size as
        bounding box.
    **image** : (H, J) ndarray
        Sliced binary region image which has the same size as bounding box.
    **inertia_tensor** : ndarray
        Inertia tensor of the region for the rotation around its mass.
    **inertia_tensor_eigvals** : tuple
        The eigenvalues of the inertia tensor in decreasing order.
    **intensity_image** : ndarray
        Image inside region bounding box.
    **label** : int
        The label in the labeled input image.
    **local_centroid** : array
        Centroid coordinate tuple ``(row, col)``, relative to region bounding
        box.
    **major_axis_length** : float
        The length of the major axis of the ellipse that has the same
        normalized second central moments as the region.
    **max_intensity** : float
        Value with the greatest intensity in the region.
    **mean_intensity** : float
        Value with the mean intensity in the region.
    **min_intensity** : float
        Value with the least intensity in the region.
    **minor_axis_length** : float
        The length of the minor axis of the ellipse that has the same
        normalized second central moments as the region.
    **moments** : (3, 3) ndarray
        Spatial moments up to 3rd order::

            m_ij = sum{ array(row, col) * row^i * col^j }

        where the sum is over the `row`, `col` coordinates of the region.
    **moments_central** : (3, 3) ndarray
        Central moments (translation invariant) up to 3rd order::

            mu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }

        where the sum is over the `row`, `col` coordinates of the region,
        and `row_c` and `col_c` are the coordinates of the region's centroid.
    **moments_hu** : tuple
        Hu moments (translation, scale and rotation invariant).
    **moments_normalized** : (3, 3) ndarray
        Normalized moments (translation and scale invariant) up to 3rd order::

            nu_ij = mu_ij / m_00^[(i+j)/2 + 1]

        where `m_00` is the zeroth spatial moment.
    **orientation** : float
        Angle between the 0th axis (rows) and the major
        axis of the ellipse that has the same second moments as the region,
        ranging from `-pi/2` to `pi/2` counter-clockwise.
    **perimeter** : float
        Perimeter of object which approximates the contour as a line
        through the centers of border pixels using a 4-connectivity.
    **slice** : tuple of slices
        A slice to extract the object from the source image.
    **solidity** : float
        Ratio of pixels in the region to pixels of the convex hull image.
    **weighted_centroid** : array
        Centroid coordinate tuple ``(row, col)`` weighted with intensity
        image.
    **weighted_local_centroid** : array
        Centroid coordinate tuple ``(row, col)``, relative to region bounding
        box, weighted with intensity image.
    **weighted_moments** : (3, 3) ndarray
        Spatial moments of intensity image up to 3rd order::

            wm_ij = sum{ array(row, col) * row^i * col^j }

        where the sum is over the `row`, `col` coordinates of the region.
    **weighted_moments_central** : (3, 3) ndarray
        Central moments (translation invariant) of intensity image up to
        3rd order::

            wmu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }

        where the sum is over the `row`, `col` coordinates of the region,
        and `row_c` and `col_c` are the coordinates of the region's weighted
        centroid.
    **weighted_moments_hu** : tuple
        Hu moments (translation, scale and rotation invariant) of intensity
        image.
    **weighted_moments_normalized** : (3, 3) ndarray
        Normalized moments (translation and scale invariant) of intensity
        image up to 3rd order::

            wnu_ij = wmu_ij / wm_00^[(i+j)/2 + 1]

        where ``wm_00`` is the zeroth spatial moment (intensity-weighted area).

    Each region also supports iteration, so that you can do::

      for prop in region:
          print(prop, region[prop])

    See Also
    --------
    label

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] https://en.wikipedia.org/wiki/Image_moment

    Examples
    --------
    >>> from skimage import data, util
    >>> from skimage.measure import label
    >>> img = util.img_as_ubyte(data.coins()) > 110
    >>> label_img = label(img, connectivity=img.ndim)
    >>> props = regionprops(label_img)
    >>> # centroid of first labeled object
    >>> props[0].centroid
    (22.72987986048314, 81.91228523446583)
    >>> # centroid of first labeled object
    >>> props[0]['centroid']
    (22.72987986048314, 81.91228523446583)

    """

    if label_image.ndim not in (2, 3):
        raise TypeError('Only 2-D and 3-D images supported.')

    if not np.issubdtype(label_image.dtype, np.integer):
        raise TypeError('Label image must be of integer type.')

    if coordinates is not None:
        if coordinates == 'rc':
            msg = ('The coordinates keyword argument to skimage.measure.'
                   'regionprops is deprecated. All features are now computed '
                   'in rc (row-column) coordinates. Please remove '
                   '`coordinates="rc"` from all calls to regionprops before '
                   'updating scikit-image.')
            warn(msg, stacklevel=2, category=FutureWarning)
        else:
            msg = ('Values other than "rc" for the "coordinates" argument '
                   'to skimage.measure.regionprops are no longer supported. '
                   'You should update your code to use "rc" coordinates and '
                   'stop using the "coordinates" argument, or use skimage '
                   'version 0.15.x or earlier.')
            raise ValueError(msg)

    regions = []

    objects = ndi.find_objects(label_image)
    for i, sl in enumerate(objects):
        if sl is None:
            continue

        label = i + 1

        props = RegionProperties(sl, label, label_image, intensity_image,
                                 cache)
        regions.append(props)

    return regions


def perimeter(image, neighbourhood=4):
    """Calculate total perimeter of all objects in binary image.

    Parameters
    ----------
    image : (N, M) ndarray
        2D binary image.
    neighbourhood : 4 or 8, optional
        Neighborhood connectivity for border pixel determination. It is used to
        compute the contour. A higher neighbourhood widens the border on which
        the perimeter is computed.

    Returns
    -------
    perimeter : float
        Total perimeter of all objects in binary image.

    References
    ----------
    .. [1] K. Benkrid, D. Crookes. Design and FPGA Implementation of
           a Perimeter Estimator. The Queen's University of Belfast.
           http://www.cs.qub.ac.uk/~d.crookes/webpubs/papers/perimeter.doc

    Examples
    --------
    >>> from skimage import data, util
    >>> from skimage.measure import label
    >>> # coins image (binary)
    >>> img_coins = data.coins() > 110
    >>> # total perimeter of all objects in the image
    >>> perimeter(img_coins, neighbourhood=4)  # doctest: +ELLIPSIS
    7796.867...
    >>> perimeter(img_coins, neighbourhood=8)  # doctest: +ELLIPSIS
    8806.268...

    """
    if image.ndim != 2:
        raise NotImplementedError('`perimeter` supports 2D images only')

    if neighbourhood == 4:
        strel = STREL_4
    else:
        strel = STREL_8
    image = image.astype(np.uint8)
    eroded_image = ndi.binary_erosion(image, strel, border_value=0)
    border_image = image - eroded_image

    perimeter_weights = np.zeros(50, dtype=np.double)
    perimeter_weights[[5, 7, 15, 17, 25, 27]] = 1
    perimeter_weights[[21, 33]] = sqrt(2)
    perimeter_weights[[13, 23]] = (1 + sqrt(2)) / 2

    perimeter_image = ndi.convolve(border_image, np.array([[10, 2, 10],
                                                           [ 2, 1,  2],
                                                           [10, 2, 10]]),
                                   mode='constant', cval=0)

    # You can also write
    # return perimeter_weights[perimeter_image].sum()
    # but that was measured as taking much longer than bincount + np.dot (5x
    # as much time)
    perimeter_histogram = np.bincount(perimeter_image.ravel(), minlength=50)
    total_perimeter = perimeter_histogram @ perimeter_weights
    return total_perimeter


def _parse_docs():
    import re
    import textwrap

    doc = regionprops.__doc__ or ''
    matches = re.finditer(r'\*\*(\w+)\*\* \:.*?\n(.*?)(?=\n    [\*\S]+)',
                          doc, flags=re.DOTALL)
    prop_doc = {m.group(1): textwrap.dedent(m.group(2)) for m in matches}

    return prop_doc


def _install_properties_docs():
    prop_doc = _parse_docs()

    for p in [member for member in dir(RegionProperties)
              if not member.startswith('_')]:
        getattr(RegionProperties, p).__doc__ = prop_doc[p]


if __debug__:
    # don't install docstrings when in optimized/non-debug mode
    _install_properties_docs()
