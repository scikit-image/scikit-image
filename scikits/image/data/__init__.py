"""Convinience functions to get sample data"""

import os

from ..io import imread
from ...image import data_dir

def camera():
    """Example gray "camera" image, often used for segmentation
    and denoising examples."""

    return imread(os.path.join(data_dir, "camera.png"))

def lena():
    """Example "Lena" image. """
    return imread(os.path.join(data_dir, "lena.png"))

def checkerboard():
    """Checkerboard image"""
    return imread(os.path.join(data_dir, "chessboard_RGB.png"))

def checkerboard_gray():
    """Checkerboard image, only gray channel"""
    return imread(os.path.join(data_dir, "chessboard_GRAY.png"))
