from ._quickshift import quickshift as _quickshift

def quickshift(image, ratio=1., kernel_size=5, max_dist=10,
               return_tree=False, sigma=0, convert2lab=True, random_seed=None):
    """Segments image using quickshift clustering in Color-(x,y) space.

    Produces an oversegmentation of the image using the quickshift mode-seeking
    algorithm.

    Parameters
    ----------
    image : (width, height, channels) ndarray
        Input image.
    ratio : float, optional, between 0 and 1 (default 1).
        Balances color-space proximity and image-space proximity.
        Higher values give more weight to color-space.
    kernel_size : float, optional (default 5)
        Width of Gaussian kernel used in smoothing the
        sample density. Higher means fewer clusters.
    max_dist : float, optional (default 10)
        Cut-off point for data distances.
        Higher means fewer clusters.
    return_tree : bool, optional (default False)
        Whether to return the full segmentation hierarchy tree and distances.
    sigma : float, optional (default 0)
        Width for Gaussian smoothing as preprocessing. Zero means no smoothing.
    convert2lab : bool, optional (default True)
        Whether the input should be converted to Lab colorspace prior to
        segmentation. For this purpose, the input is assumed to be RGB.
    random_seed : None (default) or int, optional
        Random seed used for breaking ties.

    Returns
    -------
    segment_mask : (width, height) ndarray
        Integer mask indicating segment labels.

    Notes
    -----
    The authors advocate to convert the image to Lab color space prior to
    segmentation, though this is not strictly necessary. For this to work, the
    image must be given in RGB format.

    This function is a wrapper for Cython code.

    References
    ----------
    .. [1] Quick shift and kernel methods for mode seeking,
           Vedaldi, A. and Soatto, S.
           European Conference on Computer Vision, 2008


    """
    return _quickshift(image, ratio, kernel_size, max_dist, return_tree,
                       sigma, convert2lab, random_seed)
