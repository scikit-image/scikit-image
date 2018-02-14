#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions for converting between data precision.

Offloads most of the work to numpy.

:author: Mark Harfouche

:license: modified BSD
"""

import numpy as np


def im2double(image):
    """Convert image to double precision."""
    return im2type(image, dtype=np.float64)


def im2int16(image):
    """Convert image to 16-bit signed integers."""
    return im2type(image, dtype=np.int16)


def im2single(image):
    """Convert image to single precision (float32)."""
    return im2type(image, dtype=np.float32)


def im2uint16(image):
    """Convert image to 16-bit unsigned integers."""
    return im2type(image, np.uint16)


def im2uint8(image):
    """Convert image to 8-bit unsigned integers."""
    return im2type(image, np.uint8)


def im2integer(image, dtype):
    """Covert an image to a generic integer type."""
    if not issubclass(dtype, np.integer):
        raise TypeError("dtype must be a numpy integer.")

    if issubclass(image.dtype.type, np.floating):
        # I really wanted to do something smart like:
        # image_float = np.ldexp(image, bits_output) -1
        # But that just gives you off by 1 errors everywhere
        image_float = image * np.iinfo(dtype).max
        return image_float.astype(dtype)
    else:

        bits_output = np.iinfo(dtype).bits

        if issubclass(dtype, np.signedinteger):
            bits_output = bits_output - 1

        bits_input = np.iinfo(image.dtype.type).bits
        if issubclass(image.dtype.type, np.signedinteger):
            bits_input = bits_input - 1


        if bits_output < bits_input:
            return (image // (2 ** (bits_input - bits_output))).astype(dtype)
        elif bits_output > bits_input:
            return image.astype(dtype) * (2 ** (bits_output - bits_input))
        else:
            return image.astype(dtype)


def im2float(image, dtype):
    """Convert image to a generic type of float."""
    if not issubclass(dtype, np.floating):
        raise TypeError("dtype must be a numpy float.")

    image_float = image.astype(dtype)

    if issubclass(image.dtype.type, np.integer):
        max_val = np.iinfo(image.dtype.type).max
        image_float = image_float / max_val

    return image_float


def im2type(image, dtype):
    """Convert image to a generic numpy type."""
    if issubclass(dtype, np.floating):
        return im2float(image, dtype)
    elif issubclass(dtype, np.integer):
        return im2integer(image, dtype)
    else:
        raise TypeError("Unknown dtype. Must be subclass of "
                        "np.floating or np.integer.")
