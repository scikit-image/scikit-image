#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
from scipy import ndimage
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport sqrt, atan2, fabs, fmin, fmax

from skimage.morphology import convex_hull_image


__all__ = ['regionprops']


STREL_8 = np.ones((3, 3), 'int8')
cdef float PI = 3.14159265
cdef tuple PROPS = (
    'Area',
    'BoundingBox',
    'CentralMoments',
    'Centroid',
    'ConvexArea',
#    'ConvexHull',
    'ConvexImage',
    'Eccentricity',
    'EquivDiameter',
    'EulerNumber',
    'Extent',
#    'Extrema',
    'FilledArea',
    'FilledImage',
    'HuMoments',
    'Image',
    'MajorAxisLength',
    'MinorAxisLength',
    'Moments',
    'NormalizedMoments',
    'Orientation',
#    'Perimeter',
#    'PixelIdxList',
#    'PixelList',
    'Solidity',
#    'SubarrayIdx'
)


def _central_moments(np.ndarray[np.uint8_t, ndim=2] array, double cr, double cc,
                     int order):
    cdef int p, q, r, c
    cdef np.ndarray[np.double_t, ndim=2] mu
    mu = np.zeros((order + 1, order + 1), 'double')
    for p in range(order + 1):
        for q in range(order + 1):
            for r in range(array.shape[0]):
                for c in range(array.shape[1]):
                    mu[p,q] += array[r,c] * (r - cr) ** q * (c - cc) ** p
    return mu

def _normalized_moments(np.ndarray[np.double_t, ndim=2] mu, int order):
    cdef int p, q
    cdef np.ndarray[np.double_t, ndim=2] nu
    nu = np.zeros((order + 1, order + 1), 'double')
    for p in range(order + 1):
        for q in range(order + 1):
            if p + q >= 2:
                nu[p,q] = mu[p,q] / mu[0,0]**(<double>(p + q) / 2 + 1)
            else:
                nu[p,q] = np.nan
    return nu

def _hu_moments(np.ndarray[np.double_t, ndim=2] nu):
    cdef np.ndarray[np.double_t, ndim=1] hu = np.zeros((7,), 'double')
    cdef double t0 = nu[3,0] + nu[1,2]
    cdef double t1 = nu[2,1] + nu[0,3]
    cdef double q0 = t0 * t0
    cdef double q1 = t1 * t1
    cdef double n4 = 4 * nu[1,1]
    cdef double s = nu[2,0] + nu[0,2]
    cdef double d = nu[2,0] - nu[0,2]
    hu[0] = s
    hu[1] = d * d + n4 * nu[1,1]
    hu[3] = q0 + q1
    hu[5] = d * (q0 - q1) + n4 * t0 * t1
    t0 *= q0 - 3 * q1
    t1 *= 3 * q0 - q1
    q0 = nu[3,0]- 3 * nu[1,2]
    q1 = 3 * nu[2,1] - nu[0,3]
    hu[2] = q0 * q0 + q1 * q1
    hu[4] = q0 * t0 + q1 * t1
    hu[6] = q1 * t0 - q0 * t1
    return hu

def regionprops(image, properties='all'):
    """Measure properties of labeled image regions.

    Parameters
    ----------
    image : NxM ndarray
        Labelled input image.
    properties : {'all', list, tuple}
        Shape measurements to be determined for each labeled image region.
        Default is 'all'. The following properties can be determined:
        * Area : int
           Number of pixels of region.
        * BoundingBox : tuple
           Bounding box `(min_row, min_col, max_row, max_col)`
        * CentralMoments : 3x3 ndarray
            Central moments (translation invariant) up to 3rd order.
            .. math::
                \texttt{mu} _{ji} = \sum _{x,y} \left (\texttt{array} (x,y) \\
                    \cdot (x - \bar{x} )^j \cdot (y - \bar{y} )^i \right)
        * Centroid : array
            Centroid coordinate tuple `(row, col)`.
        * ConvexArea : int
            Number of pixels of convex hull image.
        * ConvexImage : HxJ ndarray
            Convex hull image which has the same size as bounding box.
        * Eccentricity : float
            Linear eccentricity of the ellipse that has the same second-moments
            as the region (0 <= eccentricity <= 1).
        * EquivDiameter : float
            The diameter of a circle with the same area as the region.
        * EulerNumber : int
            Euler number of region. Computed as number of objects (= 1)
            subtracted by number of holes (8-connectivity).
        * Extent : float
            Ratio of pixels in the region to pixels in the total bounding box.
            Computed as `Area / (rows*cols)`
        * FilledArea : int
            Number of pixels of filled region.
        * FilledImage : HxJ ndarray
            Region image with filled holes which has the same size as bounding
            box.
        * HuMoments : tuple
            Hu moments (translation, scale and rotation invariant).
        * Image : HxJ ndarray
            Sliced region image which has the same size as bounding box.
        * MajorAxisLength : float
            The length of the major axis of the ellipse that has the same
            normalized second central moments as the region.
        * MinorAxisLength : float
            The length of the minor axis of the ellipse that has the same
            normalized second central moments as the region.
        * Moments 3x3 ndarray
            Spatial moments up to 3rd order.
            .. math::
                \texttt{m} _{ji}= \sum _{x,y} \left (\texttt{array} (x,y) \\
                    \cdot x^j \cdot y^i \right)
        * NormalizedMoments : 3x3 ndarray
            Normalized moments (translation and scale invariant) up to 3rd
            order.
            .. math::
                \texttt{nu} _{ji} = \\
                    \frac{\texttt{mu}_{ji}}{\texttt{m}_{00}^{(i+j)/2+1}}
        * Orientation : float
            Angle between the X-axis and the major axis of the ellipse that has
            the same second-moments as the region. Ranging from `-pi/2` to
            `-pi/2` in counter-clockwise direction.
        * Solidity : float
            Ratio of pixels in the region to pixels of the convex hull image.

    Returns
    -------
    properties : list of dicts
        List containing a property dict for each region. The property dicts
        contain all the specified properties plus a 'Label' field.

    References
    ----------
    B. Jähne. Digital Image Processing. Springer-Verlag,
        Berlin-Heidelberg, 6. edition, 2005.
    T. H. Reiss. Recognizing Planar Objects Using Invariant Image Features,
        LNICS, p. 676. Springer, Berlin, 1993.
    http://en.wikipedia.org/wiki/Image_moment

    Examples
    --------
    >>> from skimage.data import coins
    >>> from skimage.morphology import label
    >>> img = coins() > 110
    >>> label_img = label(img)
    >>> props = regionprops(label_img)
    """
    cdef int i, r0, c0, label
    cdef np.ndarray[np.double_t, ndim=2] m, mu, nu
    cdef double cr, cc, a, b, c

    if not np.issubdtype(image.dtype, 'int'):
        raise TypeError('labelled image must be of integer dtype')

    # determine all properties if nothing specified
    if properties == 'all':
        properties = PROPS

    props = []

    objects = ndimage.find_objects(image)
    for i, sl in enumerate(objects):
        label = i + 1

        # create property dict for current label
        obj_props = {}
        props.append(obj_props)

        obj_props['Label'] = label

        # binary image of i-th label, converting to uint8 because Cython
        # does not have support for bool dtype
        array = (image[sl] == label).astype('uint8')

        # upper left corner of object bbox
        r0 = sl[0].start
        c0 = sl[1].start

        m = _central_moments(array, 0, 0, 3)
        # centroid
        cr = m[0,1] / m[0,0]
        cc = m[1,0] / m[0,0]
        mu = _central_moments(array, cr, cc, 3)
        nu = _normalized_moments(mu, 3)

        # elements of second order central moment covariance matrix
        a = mu[2,0] / mu[0,0]
        b = mu[1,1] / mu[0,0]
        c = mu[0,2] / mu[0,0]
        # eigenvalues of covariance matrix
        l1 = fabs(0.5 * (a + c - sqrt((a - c) ** 2 + 4 * b ** 2)))
        l2 = fabs(0.5 * (a + c + sqrt((a - c) ** 2 + 4 * b ** 2)))

        # cached results which are used by several properties
        _filled_image = None
        _convex_image = None

        if 'Area' in properties:
            obj_props['Area'] = m[0,0]

        if 'BoundingBox' in properties:
            obj_props['BoundingBox'] = (r0, c0, sl[0].stop, sl[1].stop)

        if 'Centroid' in properties:
            obj_props['Centroid'] = cr + r0, cc + c0

        if 'CentralMoments' in properties:
            obj_props['CentralMoments'] = mu

        if 'ConvexArea' in properties:
            if _convex_image is None:
                _convex_image = convex_hull_image(array)
            obj_props['ConvexArea'] = np.sum(_convex_image)

        if 'ConvexImage' in properties:
            if _convex_image is None:
                _convex_image = convex_hull_image(array)
            obj_props['ConvexImage'] = _convex_image

        if 'Eccentricity' in properties:
            obj_props['Eccentricity'] = \
                sqrt(1 - (fmin(l1, l2) / fmax(l1, l2)) ** 2)

        if 'EquivDiameter' in properties:
            obj_props['EquivDiameter'] = sqrt(4 * m[0,0] / PI)

        if 'EulerNumber' in properties:
            if _filled_image is None:
                _filled_image = ndimage.binary_fill_holes(array, STREL_8)
            euler_array = _filled_image != array
            _, num = ndimage.label(euler_array, STREL_8)
            obj_props['EulerNumber'] = 1 - num

        if 'Extent' in properties:
            obj_props['Extent'] = m[0,0] / (array.shape[0] * array.shape[1])

        if 'HuMoments' in properties:
            obj_props['HuMoments'] = _hu_moments(nu)

        if 'Image' in properties:
            obj_props['Image'] = array

        if 'FilledArea' in properties:
            if _filled_image is None:
                _filled_image = ndimage.binary_fill_holes(array, STREL_8)
            obj_props['FilledArea'] = np.sum(_filled_image)

        if 'FilledImage' in properties:
            if _filled_image is None:
                _filled_image = ndimage.binary_fill_holes(array, STREL_8)
            obj_props['FilledImage'] = _filled_image

        if 'MinorAxisLength' in properties:
            obj_props['MinorAxisLength'] = fmin(l1, l2)

        if 'MajorAxisLength' in properties:
            obj_props['MajorAxisLength'] = fmax(l1, l2)

        if 'Moments' in properties:
            obj_props['Moments'] = m

        if 'NormalizedMoments' in properties:
            obj_props['NormalizedMoments'] = nu

        if 'Orientation' in properties:
            obj_props['Orientation'] = - 0.5 * atan2(2 * b, a - c)

        if 'Solidity' in properties:
            if _convex_image is None:
                _convex_image = convex_hull_image(array)
            obj_props['Solidity'] = m[0,0] / np.sum(_convex_image)

    return props
