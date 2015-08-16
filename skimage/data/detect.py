import os as _os
from .. import data_dir


def frontal_face_cascade_xml():
    """
    Returns the file's path to the trained xml file.
    """

    return _os.path.join(data_dir, 'lbpcascade_frontalface_opencv.xml')