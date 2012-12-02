"""Standard test images.

For more images, see

 - http://sipi.usc.edu/database/database.php

"""

import os as _os

from ..io import imread
from skimage import data_dir


def load(f):
    """Load an image file located in the data directory.

    Parameters
    ----------
    f : string
        File name.

    Returns
    -------
    img : ndarray
        Image loaded from skimage.data_dir.
    """
    return imread(_os.path.join(data_dir, f))


def camera():
    """Gray-level "camera" image.

    Often used for segmentation and denoising examples.

    """
    return load("camera.png")


def lena():
    """Colour "Lena" image.

    The standard, yet sometimes controversial Lena test image was
    scanned from the November 1972 edition of Playboy magazine.  From
    an image processing perspective, this image is useful because it
    contains smooth, textured, shaded as well as detail areas.

    """
    return load("lena.png")


def text():
    """Gray-level "text" image used for corner detection.

    Notes
    -----
    This image was downloaded from Wikipedia
    <http://en.wikipedia.org/wiki/File:Corner.png>`__.

    No known copyright restrictions, released into the public domain.

    """

    return load("text.png")


def checkerboard():
    """Checkerboard image.

    Checkerboards are often used in image calibration, since the
    corner-points are easy to locate.  Because of the many parallel
    edges, they also visualise distortions particularly well.

    """
    return load("chessboard_GRAY.png")


def coins():
    """Greek coins from Pompeii.

    This image shows several coins outlined against a gray background.
    It is especially useful in, e.g. segmentation tests, where
    individual objects need to be identified against a background.
    The background shares enough grey levels with the coins that a
    simple segmentation is not sufficient.

    Notes
    -----
    This image was downloaded from the
    `Brooklyn Museum Collection
    <http://www.brooklynmuseum.org/opencollection/archives/image/617/image>`__.

    No known copyright restrictions.

    """
    return load("coins.png")


def moon():
    """Surface of the moon.

    This low-contrast image of the surface of the moon is useful for
    illustrating histogram equalization and contrast stretching.

    """
    return load("moon.png")


def page():
    """Scanned page.

    This image of printed text is useful for demonstrations requiring uneven
    background illumination.

    """
    return load("page.png")
