import os as _os
from .. import data_dir


def load_file(f):
    """Load a file located in the data directory.

    Parameters
    ----------
    f : string
        File name.

    Returns
    -------
    file : file object
        File loaded from skimage.data_dir.
    """

    return open(_os.path.join(data_dir, f))


def face_cascade_detector():

    return load_file('lbpcascade_frontalface_opencv.xml')