import os as _os
data_dir = _os.path.abspath(_os.path.dirname(__file__))


def frontal_face_cascade_xml():
    """
    Returns the file's path to the trained xml file for frontal face detection
    that was taken from OpenCV repository [1]_.

    References
    ----------
    .. [1] OpenCV lbpcascade trained files
           https://github.com/Itseez/opencv/tree/master/data/lbpcascades
    """

    return _os.path.join(data_dir, 'lbpcascade_frontalface_opencv.xml')
