# coding: utf-8
from math import sqrt, atan2, pi as PI
import numpy as np
from scipy import ndimage

from skimage.morphology import convex_hull_image
from . import _moments


__all__ = ['regionprops']


STREL_4 = np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]])
STREL_8 = np.ones((3, 3), 'int8')
PROPS = (
    'Area',
    'BoundingBox',
    'CentralMoments',
    'Centroid',
    'ConvexArea',
#    'ConvexHull',
    'ConvexImage',
    'Coordinates',
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
    'MaxIntensity',
    'MeanIntensity',
    'MinIntensity',
    'MinorAxisLength',
    'Moments',
    'NormalizedMoments',
    'Orientation',
    'Perimeter',
#    'PixelIdxList',
#    'PixelList',
    'Solidity',
#    'SubarrayIdx'
    'WeightedCentralMoments',
    'WeightedCentroid',
    'WeightedHuMoments',
    'WeightedMoments',
    'WeightedNormalizedMoments'
)


def regionprops(label_image, properties=['Area', 'Centroid'],
                intensity_image=None):
    """Measure properties of labelled image regions.

    Parameters
    ----------
    label_image : (N, M) ndarray
        Labelled input image.
    properties : {'all', list}
        Shape measurements to be determined for each labelled image region.
        Default is `['Area', 'Centroid']`. The following properties can be
        determined:

        * Area : int
           Number of pixels of region.

        * BoundingBox : tuple
           Bounding box `(min_row, min_col, max_row, max_col)`

        * CentralMoments : (3, 3) ndarray
            Central moments (translation invariant) up to 3rd order.

                mu_ji = sum{ array(x, y) * (x - x_c)^j * (y - y_c)^i }

            where the sum is over the `x`, `y` coordinates of the region,
            and `x_c` and `y_c` are the coordinates of the region's centroid.

        * Centroid : array
            Centroid coordinate tuple `(row, col)`.

        * ConvexArea : int
            Number of pixels of convex hull image.

        * ConvexImage : (H, J) ndarray
            Binary convex hull image which has the same size as bounding box.

        * Coordinates : (N, 2) ndarray
            Coordinate list `(row, col)` of the region.

        * Eccentricity : float
            Eccentricity of the ellipse that has the same second-moments as the
            region. The eccentricity is the ratio of the distance between its
            minor and major axis length. The value is between 0 and 1.

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

        * FilledImage : (H, J) ndarray
            Binary region image with filled holes which has the same size as
            bounding box.

        * HuMoments : tuple
            Hu moments (translation, scale and rotation invariant).

        * Image : (H, J) ndarray
            Sliced binary region image which has the same size as bounding box.

        * MajorAxisLength : float
            The length of the major axis of the ellipse that has the same
            normalized second central moments as the region.

        * MaxIntensity: float
            Value with the greatest intensity in the region.

        * MeanIntensity: float
            Value with the mean intensity in the region.

        * MinIntensity: float
            Value with the least intensity in the region.

        * MinorAxisLength : float
            The length of the minor axis of the ellipse that has the same
            normalized second central moments as the region.

        * Moments : (3, 3) ndarray
            Spatial moments up to 3rd order.

                m_ji = sum{ array(x, y) * x^j * y^i }

            where the sum is over the `x`, `y` coordinates of the region.

        * NormalizedMoments : (3, 3) ndarray
            Normalized moments (translation and scale invariant) up to 3rd
            order.

                nu_ji = mu_ji / m_00^[(i+j)/2 + 1]

            where `m_00` is the zeroth spatial moment.

        * Orientation : float
            Angle between the X-axis and the major axis of the ellipse that has
            the same second-moments as the region. Ranging from `-pi/2` to
            `pi/2` in counter-clockwise direction.

        * Perimeter : float
            Perimeter of object which approximates the contour as a line
            through the centers of border pixels using a 4-connectivity.

        * Solidity : float
            Ratio of pixels in the region to pixels of the convex hull image.

        * WeightedCentralMoments : (3, 3) ndarray
            Central moments (translation invariant) of intensity image up to
            3rd order.

                wmu_ji = sum{ array(x, y) * (x - x_c)^j * (y - y_c)^i }

            where the sum is over the `x`, `y` coordinates of the region,
            and `x_c` and `y_c` are the coordinates of the region's centroid.

        * WeightedCentroid : array
            Centroid coordinate tuple `(row, col)` weighted with intensity
            image.

        * WeightedHuMoments : tuple
            Hu moments (translation, scale and rotation invariant) of intensity
            image.

        * WeightedMoments : (3, 3) ndarray
            Spatial moments of intensity image up to 3rd order.

                wm_ji = sum{ array(x, y) * x^j * y^i }

            where the sum is over the `x`, `y` coordinates of the region.

        * WeightedNormalizedMoments : (3, 3) ndarray
            Normalized moments (translation and scale invariant) of intensity
            image up to 3rd order.

                wnu_ji = wmu_ji / wm_00^[(i+j)/2 + 1]

            where `wm_00` is the zeroth spatial moment (intensity-weighted
            area).

    intensity_image : (N, M) ndarray, optional
        Intensity image with same size as labelled image. Default is None.

    Returns
    -------
    properties : list of dicts
        List containing a property dict for each region. The property dicts
        contain all the specified properties plus a 'Label' field.

    References
    ----------
    .. [1] Wilhelm Burger, Mark Burge. Principles of Digital Image Processing:
           Core Algorithms. Springer-Verlag, London, 2009.
    .. [2] B. Jähne. Digital Image Processing. Springer-Verlag,
           Berlin-Heidelberg, 6. edition, 2005.
    .. [3] T. H. Reiss. Recognizing Planar Objects Using Invariant Image
           Features, from Lecture notes in computer science, p. 676. Springer,
           Berlin, 1993.
    .. [4] http://en.wikipedia.org/wiki/Image_moment

    Examples
    --------
    >>> from skimage.data import coins
    >>> from skimage.morphology import label
    >>> img = coins() > 110
    >>> label_img = label(img)
    >>> props = regionprops(label_img)
    >>> props[0]['Centroid'] # centroid of first labelled object
    """
    if not np.issubdtype(label_image.dtype, 'int'):
        raise TypeError('labelled image must be of integer dtype')

    # determine all properties if nothing specified
    if properties == 'all':
        properties = PROPS

    props = []

    objects = ndimage.find_objects(label_image)
    for i, sl in enumerate(objects):
        label = i + 1

        # create property dict for current label
        obj_props = {}
        props.append(obj_props)

        obj_props['Label'] = label

        array = (label_image[sl] == label).astype('double')

        # upper left corner of object bbox
        r0 = sl[0].start
        c0 = sl[1].start

        m = _moments.central_moments(array, 0, 0, 3)
        # centroid
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]
        mu = _moments.central_moments(array, cr, cc, 3)

        # elements of the inertia tensor [a b; b c]
        a = mu[2, 0] / mu[0, 0]
        b = mu[1, 1] / mu[0, 0]
        c = mu[0, 2] / mu[0, 0]
        # eigen values of inertia tensor
        l1 = (a + c) / 2 + sqrt(4 * b ** 2 + (a - c) ** 2) / 2
        l2 = (a + c) / 2 - sqrt(4 * b ** 2 + (a - c) ** 2) / 2

        # cached results which are used by several properties
        _filled_image = None
        _convex_image = None
        _nu = None

        if 'Area' in properties:
            obj_props['Area'] = m[0, 0]

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

        if 'Coordinates' in properties:
            rr, cc = np.nonzero(array)
            obj_props['Coordinates'] = np.vstack((rr + r0, cc + c0)).T

        if 'Eccentricity' in properties:
            if l1 == 0:
                obj_props['Eccentricity'] = 0
            else:
                obj_props['Eccentricity'] = sqrt(1 - l2 / l1)

        if 'EquivDiameter' in properties:
            obj_props['EquivDiameter'] = sqrt(4 * m[0, 0] / PI)

        if 'EulerNumber' in properties:
            if _filled_image is None:
                _filled_image = ndimage.binary_fill_holes(array, STREL_8)
            euler_array = _filled_image != array
            _, num = ndimage.label(euler_array, STREL_8)
            obj_props['EulerNumber'] = - num

        if 'Extent' in properties:
            obj_props['Extent'] = m[0, 0] / (array.shape[0] * array.shape[1])

        if 'HuMoments' in properties:
            if _nu is None:
                _nu = _moments.normalized_moments(mu, 3)
            obj_props['HuMoments'] = _moments.hu_moments(_nu)

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

        if 'MajorAxisLength' in properties:
            obj_props['MajorAxisLength'] = 4 * sqrt(l1)

        if 'MinorAxisLength' in properties:
            obj_props['MinorAxisLength'] = 4 * sqrt(l2)

        if 'Moments' in properties:
            obj_props['Moments'] = m

        if 'NormalizedMoments' in properties:
            if _nu is None:
                _nu = _moments.normalized_moments(mu, 3)
            obj_props['NormalizedMoments'] = _nu

        if 'Orientation' in properties:
            if a - c == 0:
                if b > 0:
                    obj_props['Orientation'] = -PI / 4.
                else:
                    obj_props['Orientation'] = PI / 4.
            else:
                obj_props['Orientation'] = - 0.5 * atan2(2 * b, (a - c))

        if 'Perimeter' in properties:
            obj_props['Perimeter'] = perimeter(array, 4)

        if 'Solidity' in properties:
            if _convex_image is None:
                _convex_image = convex_hull_image(array)
            obj_props['Solidity'] = m[0, 0] / np.sum(_convex_image)

        if intensity_image is not None:
            weighted_array = array * intensity_image[sl]

            wm = _moments.central_moments(weighted_array, 0, 0, 3)
            # weighted centroid
            wcr = wm[0, 1] / wm[0, 0]
            wcc = wm[1, 0] / wm[0, 0]
            wmu = _moments.central_moments(weighted_array, wcr, wcc, 3)

            # cached results which are used by several properties
            _wnu = None
            _vals = None

            if 'MaxIntensity' in properties:
                if _vals is None:
                    _vals = weighted_array[array.astype('bool')]
                obj_props['MaxIntensity'] = np.max(_vals)

            if 'MeanIntensity' in properties:
                if _vals is None:
                    _vals = weighted_array[array.astype('bool')]
                obj_props['MeanIntensity'] = np.mean(_vals)

            if 'MinIntensity' in properties:
                if _vals is None:
                    _vals = weighted_array[array.astype('bool')]
                obj_props['MinIntensity'] = np.min(_vals)

            if 'WeightedCentralMoments' in properties:
                obj_props['WeightedCentralMoments'] = wmu

            if 'WeightedCentroid' in properties:
                obj_props['WeightedCentroid'] = wcr + r0, wcc + c0

            if 'WeightedHuMoments' in properties:
                if _wnu is None:
                    _wnu = _moments.normalized_moments(wmu, 3)
                obj_props['WeightedHuMoments'] = _moments.hu_moments(_wnu)

            if 'WeightedMoments' in properties:
                obj_props['WeightedMoments'] = wm

            if 'WeightedNormalizedMoments' in properties:
                if _wnu is None:
                    _wnu = _moments.normalized_moments(wmu, 3)
                obj_props['WeightedNormalizedMoments'] = _wnu

    return props


def perimeter(image, neighbourhood=4):
    """Calculate total perimeter of all objects in binary image.

    Parameters
    ----------
    image : array
        binary image
    neighbourhood : 4 or 8, optional
        neighbourhood connectivity for border pixel determination, default 4

    Returns
    -------
    perimeter : float
        total perimeter of all objects in binary image

    References
    ----------
    .. [1] K. Benkrid, D. Crookes. Design and FPGA Implementation of
           a Perimeter Estimator. The Queen's University of Belfast.
           http://www.cs.qub.ac.uk/~d.crookes/webpubs/papers/perimeter.doc
    """
    if neighbourhood == 4:
        strel = STREL_4
    else:
        strel = STREL_8
    eroded_image = ndimage.binary_erosion(image, strel)
    border_image = image - eroded_image

    # perimeter contribution: corresponding values in convolved image
    perimeter_weights = {
        1:               (5, 7, 15, 17, 25, 27),
        sqrt(2):         (21, 33),
        1 + sqrt(2) / 2: (13, 23)
    }
    perimeter_image = ndimage.convolve(border_image, np.array([[10, 2, 10],
                                                               [ 2, 1,  2],
                                                               [10, 2, 10]]))
    total_perimeter = 0
    for weight, values in perimeter_weights.items():
        num_values = 0
        for value in values:
            num_values += np.sum(perimeter_image == value)
        total_perimeter += num_values * weight

    return total_perimeter
