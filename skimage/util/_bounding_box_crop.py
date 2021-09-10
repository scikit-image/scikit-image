import numpy as np

__all__ = ['bounding_box_crop']


def bounding_box_crop(image, bounding_box, axes=None, copy=False):
    """Crop an image from a bounding box.

    Parameters
    ----------
    image : ndarray
        Input image to crop.
    bounding_box : list of 2-tuple (x, y) where x < y.
        The bounding box to use during cropping. Contains a 2-tuple of
        (min_val, max_val) for each axis in `axes` (as with Python slices,
        max_val is non-inclusive).
    axes : tuple, optional
        Axes corresponding to the limits defined in `bounding_box`. If None,
        all axes are included (in ascending order).
    copy : bool, optional
        If True, ensure that the output is not a view of `image`.

    Returns
    ----------
    out : ndarray
        The region of `image` corresponding to the bounding box.

    Examples
    --------
    >>> from skimage import data
    >>> from skimage.util.crop import bounding_box_crop
    >>> img = data.camera()
    >>> img.shape
    (512, 512)
    >>> cropped_img = bounding_box_crop(img, [(0, 100)])
    >>> cropped_img.shape
    (100, 512)
    >>> cropped_img = bounding_box_crop(img, [(0, 100), (0, 100)])
    >>> cropped_img.shape
    (100, 100)
    >>> cropped_img = bounding_box_crop(img, [(0, 100), (0, 75)], axes=[1, 0])
    >>> cropped_img.shape
    (75, 100)
    """

    # empty length of bounding box detected on None detected
    if not bounding_box:
        return image

    # check data isinstance of numpy array
    if not isinstance(image, np.ndarray):
        raise ValueError("data must be numpy array")

    # if not axes provided,
    # consider sequential cropping on axes
    if not axes:
        axes = list(range(len(bounding_box)))
    else:
        if len(axes) != len(set(axes)):
            raise ValueError("axes must be unique")
        if len(axes) != len(bounding_box):
            raise ValueError("axes and bounding_box must have same length")
        if not all(isinstance(a, int) for a in axes):
            raise ValueError("axes must be integer")
        if not all(a >= -image.ndim and a < image.ndim for a in axes):
            raise ValueError(f"axis {axes} is out of range for image with "
                             f"{image.ndim} dimensions.")

    slices = [slice(None)] * image.ndim
    for box, ax in zip(bounding_box, axes):
        axis_min, axis_max = box
        if axis_min > axis_max:
            raise ValueError(
                "In bounding_box, tuple should be sorted (min_val, max_val).")
        if axis_min < 0:
            raise ValueError("In bounding_box, values must be positive.")
        if axis_max > image.shape[ax]:
            raise ValueError(
                f"Bounding box {box} exceeds image dimension on axis {ax}.")
        slices[ax] = slice(axis_min, axis_max)
    slices = tuple(slices)
    
    if copy:
        return image[slices].copy()
    return image[slices]
