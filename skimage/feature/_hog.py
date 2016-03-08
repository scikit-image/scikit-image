from __future__ import division
import numpy as np
from .._shared.utils import assert_nD
from . import _hoghistogram
import warnings


def hog(image, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(3, 3), visualise=False, transform_sqrt=False,
        feature_vector=True, normalise=None):
    """Extract Histogram of Oriented Gradients (HOG) for a given image.

    Compute a Histogram of Oriented Gradients (HOG) by

        1. (optional) global image normalisation
        2. computing the gradient image in x and y
        3. computing gradient histograms
        4. normalising across blocks
        5. flattening into a feature vector

    Parameters
    ----------
    image : (M, N) ndarray
        Input image (greyscale).
    orientations : int
        Number of orientation bins.
    pixels_per_cell : 2 tuple (int, int)
        Size (in pixels) of a cell.
    cells_per_block  : 2 tuple (int,int)
        Number of cells in each block.
    visualise : bool, optional
        Also return an image of the HOG.
    transform_sqrt : bool, optional
        Apply power law compression to normalise the image before
        processing. DO NOT use this if the image contains negative
        values. Also see `notes` section below.
    feature_vector : bool, optional
        Return the data as a feature vector by calling .ravel() on the result
        just before returning.
    normalise : bool, deprecated
        The parameter is deprecated. Use `transform_sqrt` for power law
        compression. `normalise` has been deprecated.

    Returns
    -------
    newarr : ndarray
        HOG for the image as a 1D (flattened) array.
    hog_image : ndarray (if visualise=True)
        A visualisation of the HOG image.

    References
    ----------
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
      Human Detection, IEEE Computer Society Conference on Computer
      Vision and Pattern Recognition 2005 San Diego, CA, USA

    Notes
    -----
    Power law compression, also known as Gamma correction, is used to reduce
    the effects of shadowing and illumination variations. The compression makes
    the dark regions lighter. When the kwarg `transform_sqrt` is set to
    ``True``, the function computes the square root of each color channel
    and then applies the hog algorithm to the image.
    """
    image = np.atleast_2d(image)

    """
    The first stage applies an optional global image normalisation
    equalisation that is designed to reduce the influence of illumination
    effects. In practice we use gamma (power law) compression, either
    computing the square root or the log of each colour channel.
    Image texture strength is typically proportional to the local surface
    illumination so this compression helps to reduce the effects of local
    shadowing and illumination variations.
    """

    assert_nD(image, 2)

    if normalise is not None:
        raise ValueError("The normalise parameter was removed due to incorrect "
                         "behavior; it only applied a square root instead of a "
                         "true normalization. If you wish to duplicate the old "
                         "behavior, set ``transform_sqrt=True``.")

    if transform_sqrt:
        image = np.sqrt(image)

    """
    The second stage computes first order image gradients. These capture
    contour, silhouette and some texture information, while providing
    further resistance to illumination variations. The locally dominant
    colour channel is used, which provides colour invariance to a large
    extent. Variant methods may also include second order image derivatives,
    which act as primitive bar detectors - a useful feature for capturing,
    e.g. bar like structures in bicycles and limbs in humans.
    """

    if image.dtype.kind == 'u':
        # convert uint image to float
        # to avoid problems with subtracting unsigned numbers in np.diff()
        image = image.astype('float')

    gx = np.empty(image.shape, dtype=np.double)
    gx[:, 0] = 0
    gx[:, -1] = 0
    gx[:, 1:-1] = image[:, 2:] - image[:, :-2]
    gy = np.empty(image.shape, dtype=np.double)
    gy[0, :] = 0
    gy[-1, :] = 0
    gy[1:-1, :] = image[2:, :] - image[:-2, :]

    """
    The third stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in
    pose or appearance. The adopted method pools gradient orientation
    information locally in the same way as the SIFT [Lowe 2004]
    feature. The image window is divided into small spatial regions,
    called "cells". For each cell we accumulate a local 1-D histogram
    of gradient or edge orientations over all the pixels in the
    cell. This combined cell-level 1-D histogram forms the basic
    "orientation histogram" representation. Each orientation histogram
    divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the
    cell are used to vote into the orientation histogram.
    """

    sy, sx = image.shape
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y

    # compute orientations integral images
    orientation_histogram = np.zeros((n_cellsy, n_cellsx, orientations))

    _hoghistogram.hog_histograms(gx, gy, cx, cy, sx, sy, n_cellsx, n_cellsy,
                                 orientations, orientation_histogram)

    # now for each cell, compute the histogram
    hog_image = None

    if visualise:
        from .. import draw

        radius = min(cx, cy) // 2 - 1
        orientations_arr = np.arange(orientations)
        dx_arr = radius * np.cos(orientations_arr / orientations * np.pi)
        dy_arr = radius * np.sin(orientations_arr / orientations * np.pi)
        hog_image = np.zeros((sy, sx), dtype=float)
        for x in range(n_cellsx):
            for y in range(n_cellsy):
                for o, dx, dy in zip(orientations_arr, dx_arr, dy_arr):
                    centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                    rr, cc = draw.line(int(centre[0] - dx),
                                       int(centre[1] + dy),
                                       int(centre[0] + dx),
                                       int(centre[1] - dy))
                    hog_image[rr, cc] += orientation_histogram[y, x, o]

    """
    The fourth stage computes normalisation, which takes local groups of
    cells and contrast normalises their overall responses before passing
    to next stage. Normalisation introduces better invariance to illumination,
    shadowing, and edge contrast. It is performed by accumulating a measure
    of local histogram "energy" over local groups of cells that we call
    "blocks". The result is used to normalise each cell in the block.
    Typically each individual cell is shared between several blocks, but
    its normalisations are block dependent and thus different. The cell
    thus appears several times in the final output vector with different
    normalisations. This may seem redundant but it improves the performance.
    We refer to the normalised block descriptors as Histogram of Oriented
    Gradient (HOG) descriptors.
    """

    n_blocksx = (n_cellsx - bx) + 1
    n_blocksy = (n_cellsy - by) + 1
    normalised_blocks = np.zeros((n_blocksy, n_blocksx,
                                  by, bx, orientations))

    for x in range(n_blocksx):
        for y in range(n_blocksy):
            block = orientation_histogram[y:y + by, x:x + bx, :]
            eps = 1e-5
            normalised_blocks[y, x, :] = block / np.sqrt(block.sum() ** 2 + eps)

    """
    The final step collects the HOG descriptors from all blocks of a dense
    overlapping grid of blocks covering the detection window into a combined
    feature vector for use in the window classifier.
    """

    if feature_vector:
        normalised_blocks = normalised_blocks.ravel()

    if visualise:
        return normalised_blocks, hog_image
    else:
        return normalised_blocks
