import numpy as np

__all__ = ['crop']

def crop(image, bounding_box, axis=None):
    """Cropping images from a bounding box.
        Bounding_box (which is a 2-tuple (min_val, max_val) for each axis)
        and (optional) axis for corresponding axis order to bounding_box.

    Parameters
    ----------
    Image : ndarray
        Input array.
    Bounding_box : list of 2-tuple (x, y) where x < y.
        Bounding box.
    axis : tuple, optional
        Axis order for cropping.
        if provided, same legth as bounding_box.
        Default: None


    Returns
    ----------
    out : ndarray
        Cropped array.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.util.crop import crop
    >>> img = data.camera()
    >>> img.shape
    >>> cropped_img = crop(img, [(0, 100)])
    >>> cropped_img.shape
    >>> cropped_img = crop(img, [(0, 100), (0, 100)])
    >>> cropped_img.shape
    >>> cropped_img = crop(img, [(0, 100), (0, 75)], axis=[1, 0])
    >>> cropped_img.shape
    """


    # empty legth of bounding box detected on None detected
    if not bounding_box:
        return image

    # check data isinstance of numpy array
    if not isinstance(image, np.ndarray):
        raise ValueError("data must be numpy array")

    # if not axis provided,
    # consider sequential cropping on axis
    if not axis:
        axis = list(range(len(bounding_box)))
    else:
        if len(axis) != len(set(axis)):
            raise ValueError("axis must be unique")
        if len(axis) != len(bounding_box):
            raise ValueError("axis and bounding_box must have same length")
        if not all(isinstance(a, int) for a in axis):
            raise ValueError("axis must be integer")
        if not all(a >= 0 for a in axis):
            raise ValueError("axis must be positive")
        if not all(a < image.ndim for a in axis):
            raise ValueError("axis must be less than image.ndim")

    bbox_with_axis = list(zip(bounding_box, axis))
    # sort axis by decreasing
    bbox_with_axis.sort(key=lambda x: x[1], reverse=True)
    full_bbox_data = []
    for idx in range(image.ndim):
        if bbox_with_axis and bbox_with_axis[-1][1] == idx:
            bbox, _ = bbox_with_axis.pop()
            axis_min, axis_max = bbox
            if axis_min > axis_max:
                raise ValueError("In bounding_box, tuple should be sorted (min_val, max_val)")

            if axis_min < 0:
                raise ValueError("In bounding_box, values must be positive")
            if axis_max < 0:
                raise ValueError("In bounding_box, values must be positive")

            if axis_min > image.shape[idx]:
                raise ValueError("Invalid bounding_box!")
            if axis_max > image.shape[idx]:
                raise ValueError("Invalid bounding_box!")
            full_bbox_data.append(range(*bbox))
        else:
            full_bbox_data.append(range(image.shape[idx]))

    return image[np.ix_(*full_bbox_data)]
