"""
A bare-bones band ratio function for hyperspectral data.
"""

import numpy as np

def band_ratio(hypercube, band1, band2):
    """
    Compute a simple band ratio between two bands.

    Parameters:
        hypercube (ndarray): 3D hyperspectral cube (rows, cols, bands)
        band1 (int): Index of numerator band
        band2 (int): Index of denominator band

    Returns:
        ndarray: 2D ratio image (rows, cols)
    """
    return np.divide(hypercube[:, :, band1], hypercube[:, :, band2] + 1e-6)  # prevent div by zero