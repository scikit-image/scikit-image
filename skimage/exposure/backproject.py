import numpy as np
from skimage.color import rgb2hsv
from skimage.exposure import rescale_intensity
from scipy.ndimage import convolve

# kernel for final convolution
disc = np.array([
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0]], dtype=np.uint8)


def normalize_hsv(hsv):
    """It extends the Hue value to 0-179 and S&V values to 0-255

    Parameters
    ----------
    hsv : HSV image array
        HSV image array obtained as result of color.rgb2hsv()

    Returns
    -------
    out : HSV image array
        It is an uint8 image array in HSV format

    """
    hsv[:, :, 0] = hsv[:, :, 0]*179
    hsv[:, :, 1:] = hsv[:, :, 1:]*255
    return hsv.astype('uint8')


def histogram_backproject(img1, img2):
    """Return the image after backprojection of img2 on img1.

    Parameters
    ----------
    img1 : array
        Image array on which img2 is backprojected
    img2 : array
        Image array whose histogram is backprojected

    Returns
    -------
    out : Image array
        Single channel image array

    References
    ----------
    .. [1] "Indexing via color histograms", M.J.Swain & D.H.Ballard published
    in Computer Vision, 1990. Proceedings, Third International Conference on,
    pp.390-393.

    """

    # Both image should be of uint8 dtype
    assert (img1.dtype == np.uint8 and img2.dtype == np.uint8),\
        " both images should be of np.uint8 dtype "

    shape1, shape2 = img1.shape, img2.shape

    # Both images should be single or 3-channel
    assert len(shape1) == len(shape2),\
        "both images should be 1-channel or 3-channel"

    # for grayscale image take 1D histogram of intensity values
    if len(shape1) < 3:

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
    else:
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
