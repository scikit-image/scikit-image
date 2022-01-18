import numpy as np
from scipy.stats import pearsonr
from functools import wraps
from .._shared.utils import check_shape_equality

__all__ = ['pearson_r',
           'manders_colocalization_coeff',
           'manders_overlap_coeff',
           'overlap',
           'integrated_density',
           'average_integrated_density'
           ]

# check that all np.arrays passed to it have the same shape
def check_shape_equality_all(*args):
    im1 = args[0]
    for im in args[1:]:
        check_shape_equality(im1, im)
    return

def check_numpy_arr(arr, name, bool_expected=False):
    if type(arr) != np.ndarray:
        raise ValueError(f"{name} is of type {type(arr)} not nd.array of dtype boolean as expected")
    if bool_expected:
        if np.sum(np.where(arr>1,1,0))>0:
            raise ValueError(f"{name} is ndarray of dtype {arr.dtype} with non-binary values. Check if image mask was passed in as expected.")

def pcc(imgA, imgB, roi=None):
    """Calculate Pearson's Correlation Coefficient between pixel intensities in two channels.

    Parameters
    ----------
    imgA : (M, N) ndarray
        Image of channel A.
    imgB : (M, N) ndarray
        Image of channel 2 to be correlated with channel B.
        Must have same dimensions as `imgA`.
    roi : (M, N) ndarray of dtype bool, optional
        Only `imgA` and `imgB` pixels within this region of interest mask are included in the calculation.
        Must have same dimensions as `imgA`.

    Returns
    -------
    pcc : float
        Pearson's correlation coefficient of the pixel intensities between the two images, within the ROI if provided.
    p-value : float
        Two-tailed p-value.

    Notes
    -------

    Pearson's Correlation Coefficient (PCC) measures the linear correlation between the intensities of the two .
    Its value ranges from -1 for perfect linear anti-correlation and +1 for perfect linear correlation. The calculation
    of the p-value assumes that the intensities of pixels in each input image is normally distributed.

    Scipy's implementation of Pearson's correlation coefficient is used. Please refer to it for further information
    and caveats [1]_.

    .. math::
        r = \frac{\sum (A_i - m_A_i) (B_i - m_B_i)}
                 {\sqrt{\sum (A_i - m_A_i)^2 \sum (B_i - m_B_i)^2}}

    where
        :math:`A_i` is the value of the :math:`i^{th}` pixel in `imgA`
        :math:`B_i` is the value of the :math:`i^{th}` pixel in `imgB`,
        :math:`m_A_i` is the mean of the pixel values in `imgA`
        :math:`m_B_i` is the mean of the pixel values in `imgB`

    A low PCC value does not necessarily mean that there is no correlation between the two channel intensities, just
    that there is no linear correlation. You may wish to plot the pixel intensities of each of thw two channels in a 2D
    scatterplot and use Spearman's rank correlation if a non-linear correlation is visually identified [2]_.
    Also consider if you are interested in correlation or co-occurence, in which case a method involving segmentation
    masks (e.g. MCC or intersection coefficient) may be more suitable [3]_ [4]_.

    Providing the ROI of only relevant sections of the image (e.g. cells, or particular cellular compartments) and
    removing noise is important as the PCC is sensitive to these measures [3]_ [4]_.

    References
    -------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html
    .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
    .. [3] Dunn, K. W., Kamocka, M. M., & McDonald, J. H. (2011). A practical guide to evaluating colocalization in
           biological microscopy. American journal of physiology. Cell physiology, 300(4), C723–C742.
           https://doi.org/10.1152/ajpcell.00462.2010
    .. [4] Bolte, S. and Cordelières, F.P. (2006), A guided tour into subcellular colocalization analysis in light
           microscopy. Journal of Microscopy, 224: 213-232. https://doi.org/10.1111/j.1365-2818.2006.01706.x
    """
    if roi is None:
        roi = np.ones_like(imgA)
    check_numpy_arr(imgA, 'imgA', bool_expected=False)
    check_numpy_arr(imgB, 'imgB', bool_expected=False)
    check_numpy_arr(roi, 'roi', bool_expected=True)
    check_shape_equality_all(imgA, imgB, roi)

    imgA_masked = imgA[roi.astype(bool)]
    imgB_masked = imgB[roi.astype(bool)]
    return pearsonr(imgA_masked, imgB_masked)

def mcc(imgA, imgB_mask, roi=None):
    """Manders' colocalization coefficient between two channels.

    Parameters
    ----------
    imgA : (M, N) ndarray
        Image of channel A.
    imgB_mask : (M, N) ndarray of dytpe bool or a float
        Binary mask with segmented regions of interest in channel B.
        Must have same dimensions as `imgA`.
    roi : (M, N) ndarray of dtype bool, optional
        Only `imgA` pixel values within this region of interest mask are included in the calculation.
        Must have same dimensions as `imgA`.

    Returns
    -------
    mcc : float
        Manders' colocalization coefficient.

    Notes
    -------
    Manders' Colocalization Coefficient (MCC) is the fraction of total intensity of a certain channel that is within the
    segmented region of a second channel [1]_. It ranges from 0 for no colocalisation to 1 for complete colocalization.
    It is also referred to as M1 and M2.

    MCC is commonly used to measure the colocalization of a particular protein in a subceullar compartment.
    Typically a threshold on the second channel is set and values above this are used in the MCC calculation.In this
    implementation, the thresholded image is provided as the argument `imgB_mask`. Alternative segmentation methods on
    the second channel can also be performed, with the resulting image mask being provided as `imgB_mask`.

    The implemented equation is:

    .. math::
        r = \frac{\sum A_{i,coloc}}
                 {\sum A_i}

    where
        :math:`A_i` is the value of the :math:`i^{th}` pixel in `imgA`
        :math:`A_{i,coloc} = A_i` if math:`Bmask_i > 0`
        :math:`Bmask_i` is the value of the :math:`i^{th}` pixel in `imgB_mask`

    MCC is sensitive to noise, with diffuse signal in the first channel inflating its value. Images should be
    processed to remove out of focus and background light before the MCC is calculated [2]_.

    References
    -------
    .. [1] Manders, E.M.M., Verbeek, F.J. and Aten, J.A. (1993), Measurement of co-localization
           of objects in dual-colour confocal images. Journal of Microscopy, 169: 375-382.
           https://doi.org/10.1111/j.1365-2818.1993.tb03313.x
           https://imagej.net/media/manders.pdf
    .. [2] Dunn, K. W., Kamocka, M. M., & McDonald, J. H. (2011). A practical guide to evaluating colocalization in
           biological microscopy. American journal of physiology. Cell physiology, 300(4), C723–C742.
           https://doi.org/10.1152/ajpcell.00462.2010

    """
    if roi is None:
        roi = np.ones_like(imgA)
    check_numpy_arr(imgA, 'imgA', bool_expected=False)
    check_numpy_arr(imgB_mask, 'imgB_mask', bool_expected=True)
    check_numpy_arr(roi, 'roi', bool_expected=True)
    check_shape_equality_all(imgA, imgB_mask, roi)

    imgA = imgA[roi.astype(bool)]
    imgB_mask = imgB_mask[roi.astype(bool)]
    if (np.sum(imgA)==0):
        return 0
    return np.sum(np.multiply(imgA, imgB_mask))/np.sum(imgA)

def moc(imgA, imgB, roi=None):
    """Manders' overlap coefficient

    Parameters
    ----------
    imgA : (M, N) ndarray
        Image of channel A.
    imgB : (M, N) ndarray
        Image of channel B.
        Must have same dimensions as `imgA`
    roi : (M, N) ndarray of dtype bool, optional
        Only `imgA` and `imgB` pixel values within this region of interest mask are included in the calculation.
        Must have same dimensions as `imgA`.

    Returns
    -------
    moc: float
        Manders' Overlap Coefficient of pixel intensities between the two images.

    Notes
    -------
    Manders' Overlap Coefficient (MOC) is given by the equation [1]_:

    .. math::
        r = \frac{\sum A_i B_i}
                 {\sqrt{\sum A_i^2 \sum B_i^2}}

    where
        :math:`A_i` is the value of the :math:`i^{th}` pixel in `imgA`
        :math:`B_i` is the value of the :math:`i^{th}` pixel in `imgB`

    It ranges between 0 for no colocalization and 1 for complete colocalization of all pixels.

    MOC does not take into account pixel intensities, just the fraction of pixels that have positive values for both
    channels[2]_ [3]_. Its usefulness has been criticized as it changes in response to differences in both co-occurence
    and correlation and so a particular MOC value could indicate a wide range of colocalization patterns [4]_ [5]_.

    References
    ----------
    .. [1] Manders, E.M.M., Verbeek, F.J. and Aten, J.A. (1993), Measurement of co-localization
           of objects in dual-colour confocal images. Journal of Microscopy, 169: 375-382.
           https://doi.org/10.1111/j.1365-2818.1993.tb03313.x
           https://imagej.net/media/manders.pdf
    .. [2] Dunn, K. W., Kamocka, M. M., & McDonald, J. H. (2011). A practical guide to evaluating colocalization in
           biological microscopy. American journal of physiology. Cell physiology, 300(4), C723–C742.
           https://doi.org/10.1152/ajpcell.00462.2010
    .. [3] Bolte, S. and Cordelières, F.P. (2006), A guided tour into subcellular colocalization analysis in light
           microscopy. Journal of Microscopy, 224: 213-232. https://doi.org/10.1111/j.1365-2818.2006.01
    .. [4] Adler J, Parmryd I. (2010), Quantifying colocalization by correlation: the Pearson correlation coefficient is
           superior to the Mander's overlap coefficient. Cytometry A.  Aug;77(8):733-42.
           https://doi.org/10.1002/cyto.a.20896
    .. [5] Adler, J, Parmryd, I. Quantifying colocalization: The case for discarding the Manders overlap coefficient.
           Cytometry. 2021; 99: 910– 920. https://doi.org/10.1002/cyto.a.24336

    """
    if roi is None:
        roi = np.ones_like(imgA)
    check_numpy_arr(imgA, 'imgA', bool_expected=False)
    check_numpy_arr(imgB, 'imgB', bool_expected=False)
    check_numpy_arr(roi, 'roi', bool_expected=True)
    check_shape_equality_all(imgA, imgB, roi)

    imgA = imgA[roi.astype(bool)]
    imgB = imgB[roi.astype(bool)]
    return np.sum(np.multiply(imgA, imgB))/(np.sum(np.square(imgA))*(np.sum(np.square(imgB))))**0.5

def intersection_coefficient(imgA_mask, imgB_mask, roi=None):
    """Fraction of a channel's segmented binary mask that overlaps with a second channel's segmented binary mask.

    Parameters
    ----------
    imgA_mask : (M, N) ndarray of dtype bool
        Image of channel A.
    imgB_mask : (M, N) ndarray of dtype bool
        Image of channel B.
        Must have same dimensions as `imgA_mask`.
    roi : (M, N) ndarray of dtype bool, optional
        Only `imgA_mask` and `imgB_mask` pixels within this region of interest mask are included in the calculation.
        Must have same dimensions as `imgA_mask`.

    Returns
    -------
    Intersection coefficient, float
        Fraction of `imag1_mask` that overlaps with `imgB_mask`.

    """
    if roi is None:
        roi = np.ones_like(imgA_mask)
    check_numpy_arr(imgA_mask, 'imgA_mask', bool_expected=True)
    check_numpy_arr(imgB_mask, 'imgB_mask', bool_expected=True)
    check_numpy_arr(roi, 'roi', bool_expected=True)
    check_shape_equality_all(imgA_mask, imgB_mask, roi)

    imgA_mask = imgA_mask[roi.astype(bool)]
    imgB_mask = imgB_mask[roi.astype(bool)]
    if np.count_nonzero(imgA_mask)==0: return 0
    return np.count_nonzero(np.logical_and(imgA_mask, imgB_mask))/np.count_nonzero(imgA_mask)

def pixel_intensity_sum(img, roi=None):
    """Sum of all intensity values of image within the ROI.

    Parameters
    ----------
    img : (M, N) ndarray
    roi : (M, N) ndarray of dtype bool, optional
        Only `img` pixel values within this region of interest mask are included in the calculation.
        Must have same dimensions as `img`.

    Returns
    -------
    float
        Sum of all intensity values of image within the ROI.

    Notes
    -------
    Equivalent to "Integrated Density" "in ImageJ [1]_.

    References
    ----------
    .. [1] https://imagej.nih.gov/ij/docs/menus/analyze.html

    """
    if roi is None:
        roi = np.ones_like(img)
    check_numpy_arr(img, 'img', bool_expected=False)
    check_numpy_arr(roi, 'roi', bool_expected=True)
    check_shape_equality_all(img, roi)

    img = img[roi.astype(bool)]
    return np.sum(img)

def av_pixel_intensity(img, roi=None):
    """ Average intensity of all pixels within the ROI.

    Parameters
    ----------
    img : (M, N) ndarray
    roi : (M, N) ndarray of dtype bool, optional
        Only `img` pixel values within this region of interest mask are included in the calculation.
        Must have same dimensions as `img`.

    Returns
    -------
    float

    """
    if roi is None:
        roi = np.ones_like(img)
    check_numpy_arr(img, 'img', bool_expected=False)
    check_numpy_arr(roi, 'roi', bool_expected=True)
    check_shape_equality_all(img, roi)

    img = img[roi.astype(bool)]
    return np.sum(img)/np.size(img)
