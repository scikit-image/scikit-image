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
    'centroid': int,
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
    'local_centroid': int,
    'major_axis_length': float,
    'max_intensity': float,
    'mean_intensity': float,
    'min_intensity': float,
    'minor_axis_length': float,
    'moments': float,
    'moments_normalized': float,
    'orientation': float,
    'perimeter': float,
    'slice': object,
    'solidity': float,
    'weighted_moments_central': float,
    'weighted_centroid': int,
    'weighted_moments_hu': float,
    'weighted_local_centroid': int,
    'weighted_moments': int,
    'weighted_moments_normalized': float
}

PROP_VALS = set(PROPS.values())

        
def _cached_property(f):
    """Decorator for caching computationally expensive properties"""
    @wraps(f)
    def caching_wrapper(obj):
        if obj._cache_active:
            try:
                return obj._cache[f.__name__]
                # Found in cache
            except KeyError:
                # Add to cache
                obj._cache[f.__name__] = f(obj)
                return obj._cache[f.__name__]
        else:
            return f(obj)
    return caching_wrapper

def only2d(method):
    """Decorator for raising an exception if property is calculted on a non-2D image"""
    @wraps(method)
    def func2d(self, *args, **kwargs):
        if self._ndim > 2:
            raise NotImplementedError('Property %s is not implemented for '
                                      '3D images' % method.__name__)
        return method(self, *args, **kwargs)
    return func2d


class RegionProperties(object):
    """ Represents a region of an image and provides methods to calculate various properties of this region.
        
        Region objects are typically instantiated from ``regionprops``, and not directly.
        
        The region described is that labeled with a given integer ``label``, given array ``label_image``
        in which the value at each point corresponds to an integer label. The labeled region should be
        bounded by ``slice``. The original ``intensity_image`` may also be supplied for properties requiring this.
        
        For documentation of remaining initilizer parameters, see ``regionprops`` documentation.      
        
        Each region also supports iteration, so that you can do::
            for prop in region:
                print(prop, region[prop])
        
    """
    def __init__(self, slice, label, label_image, intensity_image, cache_active):
        
        if intensity_image is not None:
            if not intensity_image.shape == label_image.shape:
                raise ValueError('Label and intensity image must have the'
                                 ' same shape.')
         
        self._original_shape = label_image.shape
        #self._original_size = label_image.size

        self._label = label
        self._slice = slice
        self.slice = slice
        self._ndim = label_image.ndim

        self._intensity_image = intensity_image

        self._cache_active = cache_active
        self._cache = {}
        
        # Make slice on init so original labeled image can be de-referenced
        if cache_active:
            bounded_mask = label_image[slice] == label
            # standard numpy bool uses 1 byte per value, so packing into integers causes 8 times less memory to be used.
            self._mask_shape = bounded_mask.shape
            self._bounded_mask_packed = np.packbits(bounded_mask, axis=None)
            self._label_image = None  # original label image not set. If de-referenced elsewhere then memory usage will be reduced.
        else:
            self._bounded_mask_packed = None
            self._label_image = label_image

    @property
    def bounded_mask(self):
        """A binary mask of the image region in the bounding box.
           The size is the same as the bounding box of the region.
        """
        
        # When cache enabled, unpack and return cached mask
        if self._bounded_mask_packed is not None:
            size = np.prod(self._mask_shape)
            bounded_mask = np.unpackbits(self._bounded_mask_packed, axis=None)[:size].reshape(self._mask_shape).astype(np.bool)
        else: # Otherwise, slice on demand
            bounded_mask = self._label_image[self.slice] == self.label
        
        return bounded_mask

    @property
    def label(self):
        """Original integer label for this region"""
        return self._label

    @property
    def image(self):
        """ Same as bounded_mask. (Retined for backwards compatability) """
        return self.bounded_mask

    @property
    def full_mask(self):
        """A binary mask of the image region in the original labeled image.
           The size is the same as the original labeled image.
        """
        
        mask = np.zeros(self._original_shape).astype(bool)
        mask[self._slice] = self.bounded_mask
        return mask

    @property
    @_cached_property
    def area(self):
        """Number of pixels in the region"""
        
        # TODO: Possible speedup -- accelerate by counting bits in packed array
        return np.sum(self.bounded_mask)
    
    @property
    def bbox(self):
        """
        Returns
        -------
        A tuple of the bounding box's start coordinates for each dimension,
        followed by the end coordinates for each dimension, i.e:
            ``(min_row, min_col, max_row, max_col)``.
        Pixels belonging to the bounding box are in the half-open interval
            ``[min_row; max_row)`` and ``[min_col; max_col)``.
        """
        return tuple([self.slice[i].start for i in range(self._ndim)] +
                     [self.slice[i].stop for i in range(self._ndim)])
    
    @property
    def bbox_area(self):
        """Number of pixels in the bounding box"""
        
        # TODO: Faster
        return self.bounded_mask.size
        
    
    @property
    def centroid(self):
        """
        Centroid coordinate tuple ``(row, col)``.
        """
        return tuple(self.coords.mean(axis=0))

    @property
    @_cached_property
    def convex_area(self):
        """
        Number of pixels of convex hull image, which is the smallest convex
        polygon that encloses the region.
        Returns : ``int``
        """
        return np.sum(self.convex_image)

    @property
    @_cached_property
    def convex_image(self):
        """
        Binary convex hull image which has the same size as bounding box.
        Returns : ``(H, J) ndarray``
        """
        from ..morphology.convex_hull import convex_hull_image
        return convex_hull_image(self.bounded_mask)

    @property
    def coords(self):
        """
        Coordinate list ``(row, col)`` of the region.
        Returns : ``(N, 2) ndarray``
        """
        indices = np.nonzero(self.bounded_mask)
        return np.vstack([indices[i] + self.slice[i].start
                          for i in range(self._ndim)]).T

    @property
    @only2d
    def eccentricity(self):
        """
        Eccentricity of the ellipse that has the same second-moments as the
        region. The eccentricity is the ratio of the focal distance
        (distance between focal points) over the major axis length.
        The value is in the interval [0, 1).
        When it is 0, the ellipse becomes a circle.
        Returns : ``float``
        """
        l1, l2 = self.inertia_tensor_eigvals
        if l1 == 0:
            return 0
        return sqrt(1 - l2 / l1)

    @property
    def equivalent_diameter(self):
        """
        The diameter of a circle with the same area as the region.
        Returns : ``float``
        """
        if self._ndim == 2:
            return sqrt(4 * self.area / PI)
        elif self._ndim == 3:
            return (6 * self.area / PI) ** (1. / 3)
    
    @property
    def euler_number(self):
        """
        Euler characteristic of region. Computed as number of objects (= 1)
        subtracted by number of holes (8-connectivity).
        Returns : ``int``
        """
        euler_array = self.filled_image != self.bounded_mask
        _, num = label(euler_array, connectivity=self._ndim, return_num=True,
                       background=0)
        return -num + 1

    @property
    def extent(self):
        """
        Ratio of pixels in the region to pixels in the total bounding box.
        Computed as ``area / (rows * cols)``
        Returns : ``float``
        """
        return self.area / self.image.size

    @property
    def filled_area(self):
        """
        Number of pixels of the region with all the holes filled in. Describes
        the area of the filled_bounded_mask.
        Returns : ``int``
        """
        return np.sum(self.filled_image)

    @property
    @_cached_property
    def filled_bounded_mask(self):
        """
        Binary region image with filled holes which has the same size as
        bounding box.
        Returns : ``(H, J) ndarray``
        """
        structure = np.ones((3,) * self._ndim)
        return ndi.binary_fill_holes(self.bounded_mask, structure)
    
    @property
    def filled_image(self):
        """ Same as filled_bounded_mask """
        return self.filled_bounded_mask
    
    @property
    @_cached_property
    def inertia_tensor(self):
        """
        Inertia tensor of the region for the rotation around its mass.
        Returns ``(2, 2) ndarray``
        """
        mu = self.moments_central
        return _moments.inertia_tensor(self.bounded_mask, mu)
    
    @property
    @_cached_property
    def inertia_tensor_eigvals(self):
        """
        The two eigen values of the inertia tensor in decreasing order.
        Returns : tuple
        """
        return _moments.inertia_tensor_eigvals(self.bounded_mask,
                                               T=self.inertia_tensor)
    @property
    @_cached_property
    def intensity_image(self):
        """
        Original intensity image region inside region bounding box.
        Returns : ``ndarray``
        Raises : ``AttributeError`` if no intensity image specified
        """
        if self._intensity_image is None:
            raise AttributeError('No intensity image specified.')
        return self._intensity_image[self.slice] * self.bounded_mask

    def _intensity_image_double(self):
        return self.intensity_image.astype(np.double)

    @property
    def local_centroid(self):
        """
        Centroid coordinate tuple ``(row, col)``, relative to region bounding box.
        Returns : ``array``
        """
        M = self.moments
        return tuple(M[tuple(np.eye(self._ndim, dtype=int))] /
                     M[(0,) * self._ndim])

    @property
    def max_intensity(self):
        """
        Value with the greatest intensity in the region.
        Returns : ``float``
        """
        return np.max(self.intensity_image[self.bounded_mask])

    @property
    def mean_intensity(self):
        """
        Value with the mean intensity in the region.
        Returns : ``float``
        """
        return np.mean(self.intensity_image[self.bounded_mask])

    @property
    def min_intensity(self):
        """
        Value with the least intensity in the region.
        Returns : ``float``
        """
        return np.min(self.intensity_image[self.bounded_mask])

    @property
    def total_intensity(self):
        """
        Total intensity of all pixels in the given intensity image for the region.
        Returns : ``float``
        """
        return np.sum(self.intensity_image[self.bounded_mask])

    @property
    def major_axis_length(self):
        """
        The length of the major axis of the ellipse that has the same
        normalized second central moments as the region.
        Returns : ``float``
        """
        l1 = self.inertia_tensor_eigvals[0]
        return 4 * sqrt(l1)

    @property
    def minor_axis_length(self):
        """
        The length of the minor axis of the ellipse that has the same
        normalized second central moments as the region.
        Returns : ``float``
        """
        l2 = self.inertia_tensor_eigvals[-1]
        return 4 * sqrt(l2)

    @property
    @_cached_property
    def moments(self):
        """
        Spatial moments up to 3rd order::
            ``m_ij = sum{ array(row, col) * row^i * col^j }``
        where the sum is over the `row`, `col` coordinates of the region.
        Returns : (3, 3) ndarray
        """
        M = _moments.moments(self.bounded_mask.astype(np.uint8), 3)
        return M

    @property
    @_cached_property
    def moments_central(self):
        """
        Central moments (translation invariant) up to 3rd order::
            ``mu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }``
        where the sum is over the `row`, `col` coordinates of the region,
        and `row_c` and `col_c` are the coordinates of the region's centroid.
        Returns : ``(3, 3) ndarray``
        """
        mu = _moments.moments_central(self.bounded_mask.astype(np.uint8),
                                      self.local_centroid, order=3)
        return mu

    @property
    @only2d
    def moments_hu(self):
        """
        Hu moments (translation, scale and rotation invariant).
        Returns : ``tuple``
        """
        return _moments.moments_hu(self.moments_normalized)

    @property
    @_cached_property
    def moments_normalized(self):
        """
        Normalized moments (translation and scale invariant) up to 3rd order::
            ``nu_ij = mu_ij / m_00^[(i+j)/2 + 1]``
        where `m_00` is the zeroth spatial moment.
        Returns : ``(3, 3) ndarray``
        """
        return _moments.moments_normalized(self.moments_central, 3)

    @property
    @only2d
    def orientation(self):
        """
        Angle between the 0th axis (rows) and the major
        axis of the ellipse that has the same second moments as the region,
        ranging from `-pi/2` to `pi/2` counter-clockwise.
        Returns : ``float``
        """
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
        """
        Perimeter of object which approximates the contour as a line
        through the centers of border pixels using a 4-connectivity.
        Returns : ``float``
        """
        return perimeter(self.bounded_mask, 4)

    @property
    def solidity(self):
        """
        Ratio of pixels in the region to pixels of the convex hull image.
        Returns : ``float``
        """
        return self.area / self.convex_area

    @property
    def weighted_centroid(self):
        """
        Centroid coordinate tuple ``(row, col)`` weighted with intensity
        image.
        Returns : ``array``
        """
        ctr = self.weighted_local_centroid
        return tuple(idx + slc.start
                     for idx, slc in zip(ctr, self.slice))

    @property
    def weighted_local_centroid(self):
        """
        Centroid coordinate tuple ``(row, col)``, relative to region bounding
        box, weighted with intensity image.
        Returns : ``array``
        """
        M = self.weighted_moments
        return (M[tuple(np.eye(self._ndim, dtype=int))] /
                M[(0,) * self._ndim])

    @property
    @_cached_property
    def weighted_moments(self):
        """
        Spatial moments of intensity image up to 3rd order::
            ``wm_ij = sum{ array(row, col) * row^i * col^j }``
        where the sum is over the `row`, `col` coordinates of the region.
        Returns : ``(3, 3) ndarray``
        """
        return _moments.moments(self._intensity_image_double(), 3)

    @property
    @_cached_property
    def weighted_moments_central(self):
        """
        Central moments (translation invariant) of intensity image up to
        3rd order::
            ``wmu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }``
        where the sum is over the `row`, `col` coordinates of the region,
        and `row_c` and `col_c` are the coordinates of the region's weighted centroid.
        Returns : ``(3, 3) ndarray``
        """
        ctr = self.weighted_local_centroid
        return _moments.moments_central(self._intensity_image_double(),
                                        center=ctr, order=3)

    @property
    @only2d
    def weighted_moments_hu(self):
        """
        Hu moments (translation, scale and rotation invariant) of intensity image.
        Returns : ``tuple``
        """
        return _moments.moments_hu(self.weighted_moments_normalized)

    @property
    @_cached_property
    def weighted_moments_normalized(self):
        """
        Normalized moments (translation and scale invariant) of intensity
        image up to 3rd order::
            wnu_ij = wmu_ij / wm_00^[(i+j)/2 + 1]
        where ``wm_00`` is the zeroth spatial moment (intensity-weighted area).
        Returns : ``(3, 3) ndarray``
        """
        return _moments.moments_normalized(self.weighted_moments_central, 3)

    ## Does anyone actually use this?
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

    ## Could not find any examples of code using this way, or via the (undocumented) backwards-compatability keys
    #def __getitem__(self, key):
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


def regionprops_table(label_image, intensity_image=None, cache=True,
                      properties=('label', 'bbox'), separator='-'):
    """Find image properties and convert them into a dictionary

    Parameters
    ----------
    label_image : (N, M) ndarray
        Labeled input image. Labels with value 0 are ignored.
    intensity_image : (N, M) ndarray, optional
        Intensity (i.e., input) image with same size as labeled image.
        Default is None.
    cache : bool, optional
        Determine whether to cache calculated properties. The computation is
        much faster for cached properties, whereas the memory consumption
        increases.
    coordinates : 'rc' or 'xy', optional
        Coordinate conventions for 2D images. (Only 'rc' coordinates are
        supported for 3D images.)
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
    return _props_to_dict(regions, properties=properties, separator=separator)


def regionprops(label_image, intensity_image=None, cache=True):
    """Measure properties of labeled image regions.

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
        Setting ``cache`` to True will enable caching of the bounded regions of the labeled image region and
        caching of the calculation results for the computationally intensive properties.
        This option increases speed at the cost of allocating some memory for each Region object.
        However, in applications where the original label_image(s) are not needed after instantiating regions,
        a significant reduction in overall memory usage is possible when enabling caching. This is because
        the original label_images can now be de-referenced.

    Returns
    -------
    List of ``RegionProperties`` objects.

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
    (22.729879860483141, 81.912285234465827)
    >>> # centroid of first labeled object
    >>> props[0]['centroid']
    (22.729879860483141, 81.912285234465827)

    """

    if label_image.ndim not in (2, 3):
        raise TypeError('Only 2-D and 3-D images supported.')

    if not np.issubdtype(label_image.dtype, np.integer):
        raise TypeError('Label image must be of integer type.')

    regions = []

    objects = ndi.find_objects(label_image)
    for i, sl in enumerate(objects):
        if sl is None:
            continue

        label = i + 1

        props = RegionProperties(sl, label, label_image, intensity_image, cache)
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

