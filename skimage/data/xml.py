import os as _os
from .. import data_dir


def face_cascade_detector():
    """
    Returns the filepath to the trained xml file.
    """

    return _os.path.join(data_dir, 'lbpcascade_frontalface_opencv.xml')