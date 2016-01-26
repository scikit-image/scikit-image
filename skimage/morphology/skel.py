from __future__ import division, print_function, absolute_import

import numpy as np

from ._skel import _compute_thin_image


def _prepare_image(img_in):
    """Convert to a binary image, pad the it w/ zeros, and ensure it's 3D.
    """
    if img_in.ndim < 2 or img_in.ndim > 3:
        raise ValueError('expect 2D, got ndim = %s' % img_in.ndim)

    img = img_in.copy()
    if img.ndim == 2:
        img = img.reshape((1,) + img.shape)

    # normalize to binary
    img[img != 0] = 1

    # pad w/ zeros to simplify dealing w/ neighborhood of a pixel
    img_o = np.zeros(tuple(s + 2 for s in img.shape),
                     dtype=np.uint8)
    img_o[1:-1, 1:-1, 1:-1] = img.astype(np.uint8)
    return img_o


def _postprocess_image(img_o):
    """Clip the image (padding is an implementation detail), convert to b/w.
    """
    img_oo = img_o[1:-1, 1:-1, 1:-1]
    img_oo = img_oo.squeeze()
    img_oo *= 255
    return img_oo


def compute_thin_image(img_in):
    img = _prepare_image(img_in)
    img = _compute_thin_image(img)
    img = _postprocess_image(img)
    return img


if __name__ == "__main__":
    pass
