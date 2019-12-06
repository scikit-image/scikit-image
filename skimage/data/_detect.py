import os as _os
data_dir = _os.path.abspath(_os.path.dirname(__file__))


def lbp_frontal_face_cascade_filename():
    """
    Returns the path to the XML file containing information about the weak
    classifiers of a cascade classifier trained using LBP features. It is part
    of the OpenCV repository [1]_.

    References
    ----------
    .. [1] OpenCV lbpcascade trained files
           https://github.com/Itseez/opencv/tree/master/data/lbpcascades
    """

    return _os.path.join(data_dir, 'lbpcascade_frontalface_opencv.xml')
