import os
from .novice import picture

def open(path):
    """
    Creates a new picture object from the given image path
    """
    return picture(os.path.abspath(path))
