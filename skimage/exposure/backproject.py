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


def normalize_hsv(hsv):
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


def histogram_backproject(img1, img2, multichannel=True):
    """Project the histogram of one image onto another

    Parameters
    ----------
    img1 : (M, N, 3) ndarray or (M, N) ndarray
        Image array on which img2 is backprojected
    img2 : (M, N, 3) ndarray or (M, N) ndarray
        Image array whose histogram is backprojected
    multichannel : bool, optional
        Whether the last axis of the image is to be interpreted as multiple
        channels or another spatial dimension. Default Value : True

    Returns
    -------
    out : (M, N) ndarray
        Backprojected image.  This is a grey-level image that indicates the
        level of correspondence to the histogram of the object image.

    References
    ----------
    .. [1] Swain, Michael J., and Dana H. Ballard. "Indexing via color
           histograms." Active Perception and Robot Vision. Springer Berlin
           Heidelberg, 1992. 261-273.

    """

    # Both image should be of uint8 dtype
    if img1.dtype != np.uint8:
        img1 = img_as_ubyte(img1)
    if img2.dtype != np.uint8:
        img2 = img_as_ubyte(img2)

    isGray = False
    isColor = False

    if img1.ndim == 2 and img2.ndim == 2:
        isGray = True
    elif multichannel and img1.ndim == 3 and img2.ndim == 3:
        isColor = True
    elif not multichannel and img1.ndim >= 2 and img2.ndim >= 2:
        img1 = img1.reshape(img1.shape[:2])
        img2 = img2.reshape(img2.shape[:2])
        isGray = True
    else:
        raise ValueError("Both images should be 1-channel or 3-channel \
            and multichannel should be True for 3-channel images")

    # for grayscale image take 1D histogram of intensity values
    if isGray:

        # find histograms
        hist1 = np.bincount(img1.ravel(), minlength=256)
        hist2 = np.bincount(img2.ravel(), minlength=256)

        # find their ratio hist2/hist1
        R = np.float64(hist2) / (hist1 + 1)

        # Now apply this ratio as the palette to original image, img1
        B = R[img1]
        B = np.minimum(B, 1)
        B = rescale_intensity(B, out_range=(0, 255))
        B = np.uint8(B)
        B = convolve(B, disc)
        return B

    # if color image, take 2D histogram
    elif isColor:
        # convert images to hsv plane
        hsv_img1 = rgb2hsv(img1)
        hsv_img2 = rgb2hsv(img2)
        hsv_img1 = normalize_hsv(hsv_img1)
        hsv_img2 = normalize_hsv(hsv_img2)

        # find their color 2D histograms
        h1, s1, v1 = np.dsplit(hsv_img1, (1, 2))
        hist1, _, _ = np.histogram2d(h1.ravel(), s1.ravel(),
                                     [180, 256], [[0, 180], [0, 256]])
        h2, s2, v2 = np.dsplit(hsv_img2, (1, 2))
        hist2, _, _ = np.histogram2d(h2.ravel(), s2.ravel(),
                                     [180, 256], [[0, 180], [0, 256]])

        # find their ratio hist2/hist1
        R = hist2 / (hist1 + 1)

        # backproject
        B = R[h1.ravel(), s1.ravel()]
        B = np.minimum(B, 1)
        B = B.reshape(img1.shape[:2])
        B = rescale_intensity(B, out_range=(0, 255))
        B = convolve(B, disc)
        B = np.clip(B, 0, 255).astype('uint8')
        return B
