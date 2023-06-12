"""
Image registration module

This module provides a set of functions to register (align two images into single coordinate system) based on various transformation models. Implemented algorithms include phase_cross_correlation and optical flow estimator.

Functions
---------
phase_cross_correlation: function
Perform the phase correlation to determine the shift between two images using DFT upsampling.

optical_flow_tvl1: function
Estimates the optical flow components for each axis between two images using TV-L1 solver.

optical_flow_ilk: function
Estimates the optical flow between two images using iterative Lucas-Kanade(iLK) solver.

See Also
--------
See the documentation for each function for details.

Notes
-----
Color images are not supported for optical flow estimators.

Examples
--------
>>> from skimage import data
>>> from skimage.data import stereo_motorcycle
>>> from skimage.registration import optical_flow_tvl1
>>> image0, image1, disp =stereo_motorcycle()
>>> image0=rgb2gray(image0)
>>> image1=rgb2gray(image1)
>>> flow= optical_flow_tvl1(image1,image0)

"""

from ._optical_flow import optical_flow_tvl1, optical_flow_ilk
from ._phase_cross_correlation import phase_cross_correlation

__all__ = [
    'optical_flow_ilk',
    'optical_flow_tvl1',
    'phase_cross_correlation'
]
