import numpy as np
from skimage.color import rgb2hsv
from skimage.exposure import rescale_intensity
from scipy.ndimage import convolve

# kernel for final convolution
disc = np.array([[0, 0, 1, 0, 0],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [1, 1, 1, 1, 1],
                 [0, 0, 1, 0, 0]], dtype=np.uint8)


def _normalize_hsv(hsv):
    """It changes the range of Hue and Saturation to [0, 179] and [0, 255]
    respectively.

    Parameters
    ----------
    hsv : HSV image array
        HSV image array obtained as result of color.rgb2hsv()

    Returns
    -------
    out : HSV image array
        It is an uint8 image array in HSV format

    """

    return ([179, 255, 1] * hsv).astype(np.uint8)


def histogram_backprojection(image, template, multichannel=True):
    """Project the histogram of one image onto another

    Parameters
    ----------
    image : (M, N, 3) ndarray or (M, N) ndarray
        input image.
    template : (M, N, 3) ndarray or (M, N) ndarray
        Template for the histogram reference.
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension.

    Returns
    -------
    out : (M, N) ndarray
        Backprojected image. Greyscale image that indicates the
        level of correspondence to the histogram of the object image.

    References
    ----------
    .. [1] Swain, Michael J., and Dana H. Ballard. "Indexing via color
           histograms." Active Perception and Robot Vision. Springer Berlin
           Heidelberg, 1992. 261-273.  DOI:`10.1109/ICCV.1990.139558`

    """

    # Both image should be of uint8 dtype
    if image.dtype != np.uint8:
        image = img_as_ubyte(image)
    if template.dtype != np.uint8:
        template = img_as_ubyte(template)

    isGray = False
    isColor = False

    if image.ndim == 2 and template.ndim == 2:
        isGray = True
    elif multichannel and image.ndim == 3 and template.ndim == 3:
        isColor = True
    elif not multichannel and image.ndim >= 2 and template.ndim >= 2:
        image = image.reshape(image.shape[:2])
        template = template.reshape(template.shape[:2])
        isGray = True
    else:
        raise ValueError("Both images should be 1-channel or 3-channel \
            and multichannel should be True for 3-channel images")

    # for grayscale image take 1D histogram of intensity values
    if isGray:

        # find histograms
        hist1 = np.bincount(image.ravel(), minlength=256)
        hist2 = np.bincount(template.ravel(), minlength=256)

        # find their ratio hist2/hist1
        R = np.float64(hist2) / (hist1 + 1)

        # Now apply this ratio as the palette to original image, image
        B = R[image]
        B = np.minimum(B, 1)
        B = rescale_intensity(B, out_range=(0, 255))
        B = np.uint8(B)
        B = convolve(B, disc)
        return B

    # if color image, take 2D histogram
    elif isColor:
        # convert images to hsv plane
        hsv_image = rgb2hsv(image)
        hsv_template = rgb2hsv(template)
        hsv_image = _normalize_hsv(hsv_image)
        hsv_template = _normalize_hsv(hsv_template)

        # find their color 2D histograms
        h1, s1, v1 = np.dsplit(hsv_image, (1, 2))
        hist1, _, _ = np.histogram2d(h1.ravel(), s1.ravel(),
                                     [180, 256], [[0, 180], [0, 256]])
        h2, s2, v2 = np.dsplit(hsv_template, (1, 2))
        hist2, _, _ = np.histogram2d(h2.ravel(), s2.ravel(),
                                     [180, 256], [[0, 180], [0, 256]])

        # find their ratio hist2/hist1
        R = hist2 / (hist1 + 1)

        # backproject
        B = R[h1.ravel(), s1.ravel()]
        B = np.minimum(B, 1)
        B = B.reshape(image.shape[:2])
        B = rescale_intensity(B, out_range=(0, 255))
        B = convolve(B, disc)
        B = np.clip(B, 0, 255).astype('uint8')
        return B
