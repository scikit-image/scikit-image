"""
:author: Brian Holt, 2011
:license: modified BSD
"""

import numpy as np
from scipy import sqrt, pi, arctan, cos, sin


def hog(image, n_orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(3, 3), visualise=False, normalise=False):
    """  Extract Histogram of Oriented Gradients (HOG) for a given image.

    Compute a Histogram of Oriented Gradients (HOG) by
        1) (optional) global image normalisation
        2) computing the gradient image in x and y
        3) computing gradient histograms
        3) normalise across blocks
        4) flatten into a feature vector

    Parameters
    ----------
    image: ndarray, 2D
        2D image (greyscale)

    n_orientations  : int
        number of orientation bins

    pixels_per_cell  : 2 tuple (int,int)
        pixels per cell, size in pixels of a cell

    cells_per_block  : 2 tuple (int,int)
        cells per block, number of cells in each block

    visualise : bool, optional
        return an image of the HOG

    normalise : bool, optional
        apply power law compression to normalise the image before
        processing

    Returns
    -------
    newarr : ndarray
        HOG for the image as a 1D (flattened) array.

    hog_image : PIL Image (if visualise=True)
        A visualisation of the HOG image

    References
    ----------
    * http://en.wikipedia.org/wiki/Histogram_of_oriented_gradients

    * Dalal, N and Triggs, B, Histograms of Oriented Gradients for
      Human Detection, IEEE Computer Society Conference on Computer
      Vision and Pattern Recognition 2005 San Diego, CA, USA
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

    if image.ndim == 3:
        # replace RGB with locally dominant colour channel
        pass  # TODO
    if normalise:
        image = sqrt(image)

    """
    The second stage computes first order image gradients. These capture
    contour, silhouette and some texture information, while providing
    further resistance to illumination variations. The locally dominant
    colour channel is used, which provides colour invariance to a large
    extent. Variant methods may also include second order image derivatives,
    which act as primitive bar detectors - a useful feature for capturing,
    e.g. bar like structures in bicycles and limbs in humans.
    """

    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:-1, :-1] = image[:-1,:-1]-image[:-1,1:]
    gy[:-1, :-1] = image[:-1,:-1]-image[1:,:-1]
    #gx[:-1, :-1] = np.diff(image, n=1, axis=0)
    #gy[:-1, :-1] = np.diff(image, n=1, axis=1)

    #import Image
    #Image.fromarray(gx).show()
    #Image.fromarray(gy).show()

    """
    The third stage aims to produce an encoding that is sensitive to
    local image content while remaining resistant to small changes in pose
    or appearance. The adopted method pools gradient orientation information
    locally in the same way as the SIFT [Lowe 2004] feature. The image window
    is divided into small spatial regions, called "cells". For each cell we
    accumulate a local 1-D histogram of gradient or edge orientations over
    all the pixels in the cell. This combined cell-level 1-D histogram
    forms the basic "orientation histogram" representation. Each orientation
    histogram divides the gradient angle range into a fixed number of
    predetermined bins. The gradient magnitudes of the pixels in the cell
    are used to vote into the orientation histogram.
    """

    magnitude = sqrt(gx ** 2 + gy ** 2)
    orientation = arctan(gy / (gx + 1e-15)) * (180 / pi) + 90

    # compute n_orientations integral images
    integral_images = []
    for i in range(0, n_orientations):
        #create new integral image for this orientation
        # isolate orientations in this range

        temp_ori = np.where(orientation < 180 / n_orientations * (i + 1),
                            orientation, 0)
        temp_ori = np.where(orientation >= 180 / n_orientations * i,
                            temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, magnitude, 0)

        #compute integral image
        integral = np.cumsum(np.cumsum(temp_mag, axis=0, dtype=float),
                axis=1, dtype=float)
        integral_images.append(integral)

    sx, sy = image.shape
    cx, cy = pixels_per_cell
    bx, by = cells_per_block

    n_cellsx = int(np.floor(sx // cx))  # number of cells in x
    n_cellsy = int(np.floor(sy // cy))  # number of cells in y

    # now for each cell, compute the histogram
    orientation_histogram = np.zeros((n_cellsx, n_cellsy, n_orientations))

    radius = min(cx, cy) // 2 - 1
    hog_image = None
    if visualise:
        import Image
        import ImageDraw
        hog_image = Image.new("F", (sy, sx))
        draw = ImageDraw.Draw(hog_image)

    for x in range(0, n_cellsx):
        for y in range(0, n_cellsy):
            for o in range(0, n_orientations):
                # compute the histogram from integral image
                #print x, y, o
                A = integral_images[o][x * cx, y * cy]
                B = integral_images[o][(x + 1) * cx - 1, y * cy]
                C = integral_images[o][(x + 1) * cx - 1, (y + 1) * cy - 1]
                D = integral_images[o][x * cx, (y + 1) * cy - 1]
                orientation_histogram[x, y, o] = A + C - D - B

                if visualise:
                    centre = tuple([y * cy + cy // 2, x * cx + cx // 2])
                    dx = radius * cos(float(o) / n_orientations * np.pi)
                    dy = radius * sin(float(o) / n_orientations * np.pi)
                    draw.line([(centre[0] - dx, centre[1] - dy),
                               (centre[0] + dx, centre[1] + dy)],
                              fill=orientation_histogram[x, y, o])

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
    normalised_blocks = np.zeros((n_blocksx, n_blocksy,
                                  bx, by, n_orientations))

    for x in range(0, n_blocksx):
        for y in range(0, n_blocksy):
            block = orientation_histogram[x:x + bx, y:y + by, :]
            eps = 1e-5
            normalised_blocks[x, y, :] = block / sqrt(block.sum() ** 2 + eps)

    """
     The final step collects the HOG descriptors from all blocks of a dense
     overlapping grid of blocks covering the detection window into a combined
     feature vector for use in the window classifier
    """

    if visualise:
        return normalised_blocks.ravel(), hog_image
    else:
        return normalised_blocks.ravel()
