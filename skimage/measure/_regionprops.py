# coding: utf-8
from math import sqrt, atan, pi as PI
import numpy as np
from scipy import ndimage

from skimage.morphology import convex_hull_image
from . import _moments


__all__ = ['regionprops']


STREL_8 = np.ones((3, 3), 'int8')
PROPS = (
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


def regionprops(image, properties='all'):
    """Measure properties of labelled image regions.

    Parameters
    ----------
    image : NxM ndarray
        Labelled input image.
    properties : {'all', list, tuple}
        Shape measurements to be determined for each labelled image region.
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
            Eccentricity of the ellipse that has the same second-moments as the
            region (1 <= Eccentricity < Inf), where Eccentricity = 1 corresponds
            to a circular disk and elongated regions have Eccentricity > 1.
            .. math::
                a_1 = \mu _{2,0} + \mu _{0,2} + \sqrt{(\mu _{2,0} - \\
                    \mu _{0,2})^2 + 4 {\mu _{1,1}}^2}
                a_2 = \mu _{2,0} + \mu _{0,2} - \sqrt{(\mu _{2,0} - \\
                    \mu _{0,2})^2 + 4 {\mu _{1,1}}^2}
                Ecc = \frac{a_1}{a_2}
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
            .. math::
                a_1 = \mu _{2,0} + \mu _{0,2} + \sqrt{(\mu _{2,0} - \\
                    \mu _{0,2})^2 + 4 {\mu _{1,1}}^2}
                A = \sqrt{\frac{2 a_1}{\mu _{0,0}}}
        * MinorAxisLength : float
            The length of the minor axis of the ellipse that has the same
            normalized second central moments as the region.
            .. math::
                a_2 = \mu _{2,0} + \mu _{0,2} - \sqrt{(\mu _{2,0} - \\
                    \mu _{0,2})^2 + 4 {\mu _{1,1}}^2}
                A = \sqrt{\frac{2 a_2}{\mu _{0,0}}}
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
    Wilhelm Burger, Mark Burge. Principles of Digital Image Processing: Core
        Algorithms. Springer-Verlag, London, 2009.
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
    >>> props[0]['Centroid'] # centroid of first labelled object
    """
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

        m = _moments.central_moments(array, 0, 0, 3)
        # centroid
        cr = m[0,1] / m[0,0]
        cc = m[1,0] / m[0,0]
        mu = _moments.central_moments(array, cr, cc, 3)

        # elements of the inertia tensor
        a = mu[2,0]
        b = mu[1,1]
        c = mu[0,2]

        # cached results which are used by several properties
        _filled_image = None
        _convex_image = None
        _nu = None
        _a1 = None
        _a2 = None

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
            if _a2 is None:
                _a2 = a + c - sqrt((a - c)**2 + 4 * b ** 2)
            if _a1 is None:
                _a1 = a + c + sqrt((a - c)**2 + 4 * b ** 2)
            obj_props['Eccentricity'] = _a1 / _a2

        if 'EquivDiameter' in properties:
            obj_props['EquivDiameter'] = sqrt(4 * m[0,0] / PI)

        if 'EulerNumber' in properties:
            if _filled_image is None:
                _filled_image = ndimage.binary_fill_holes(array, STREL_8)
            euler_array = _filled_image != array
            _, num = ndimage.label(euler_array, STREL_8)
            obj_props['EulerNumber'] =  - num

        if 'Extent' in properties:
            obj_props['Extent'] = m[0,0] / (array.shape[0] * array.shape[1])

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
            if _a1 is None:
                _a1 = a + c + sqrt((a - c)**2 + 4 * b ** 2)
            obj_props['MajorAxisLength'] = sqrt(2 * _a1 / m[0,0])

        if 'MinorAxisLength' in properties:
            if _a2 is None:
                _a2 = a1 = a + c - sqrt((a - c)**2 + 4 * b ** 2)
            obj_props['MinorAxisLength'] = sqrt(2 * _a2 / m[0,0])

        if 'Moments' in properties:
            obj_props['Moments'] = m

        if 'NormalizedMoments' in properties:
            if _nu is None:
                _nu = _moments.normalized_moments(mu, 3)
            obj_props['NormalizedMoments'] = _nu

        if 'Orientation' in properties:
            obj_props['Orientation'] = - 0.5 * atan(2 * b / (a - c))

        if 'Solidity' in properties:
            if _convex_image is None:
                _convex_image = convex_hull_image(array)
            obj_props['Solidity'] = m[0,0] / np.sum(_convex_image)

    return props
