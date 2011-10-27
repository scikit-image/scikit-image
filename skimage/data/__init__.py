"""Convenience functions to load sample data.

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
    """Gray "camera" image, often used for segmentation
    and denoising examples.

    """
    return load("camera.png")

def lena():
    """Colour "Lena" image.

    """
    return load("lena.png")

def checkerboard():
    """Checkerboard image.

    """
    return load("chessboard_RGB.png")

def coins():
    """Greek coins from Pompeii.

    Notes
    -----
    This image was downloaded from the
    `Brooklyn Museum Collection
    <http://www.brooklynmuseum.org/opencollection/archives/image/617/image>`__.

    No known copyright restrictions.

    """
    return load("coins.jpg")
