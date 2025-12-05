"""Synthetic data generation."""

from skimage.data._binary_blobs import _binary_blobs_sk2_implementation

from .._shared import api


@api.copy_interface(_binary_blobs_sk2_implementation)
def binary_blobs(*args, **kwargs):
    return _binary_blobs_sk2_implementation(*args, **kwargs)
