## Copyright (C) 2005 Stefan van der Walt <stefan@sun.ac.za>
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##
##  1. Redistributions of source code must retain the above copyright
##     notice, this list of conditions and the following disclaimer.
##  2. Redistributions in binary form must reproduce the above copyright
##     notice, this list of conditions and the following disclaimer in
##     the documentation and/or other materials provided with the
##     distribution.
##
## THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
## IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
## WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
## DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
## INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
## (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
## SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
## HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
## STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
## IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
## POSSIBILITY OF SUCH DAMAGE.

__all__ = ['hough']

import numpy as np

itype = np.uint16 # See ticket 225

def hough(img, angles=None):
    """Perform a straight line Hough transform.

    Parameters
    ----------
    img : (M, N) bool ndarray
        Thresholded input image.
    angles : ndarray or list
        Angles at which to compute the transform.

    Returns
    -------
    H : 2-D ndarray
        Hough transform coefficients.
    distances : ndarray
        Distance values.
    angles : ndarray
        Angle values.

    Examples
    --------
    Generate a test image:

    >>> img = np.zeros((100, 150), dtype=bool)
    >>> img[30, :] = 1
    >>> img[:, 65] = 1
    >>> img[35:45, 35:50] = 1
    >>> for i in range(90):
    >>>     img[i, i] = 1
    >>> img += np.random.random(img.shape) > 0.95

    Apply the Hough transform:

    >>> out, angles, d = houghtf(img)

    Plot the results:

    >>> import matplotlib.pyplot as plt
    >>> plt.imshow(out, cmap=plt.cm.bone)
    >>> plt.xlabel('Angle (degree)')
    >>> plt.ylabel('Distance %d (pixel)' % d[0])
    >>> plt.show()

    """
    if img.ndim != 2:
        raise ValueError("Input must be a two-dimensional array")

    img = img.astype(bool)

    if angles is None:
        angles = np.linspace(-90,90,180)

    theta = angles / 180. * np.pi
    d = np.ceil(np.hypot(*img.shape))
    nr_bins = 2*d - 1
    bins = np.linspace(-d, d, nr_bins)
    out = np.zeros((nr_bins, len(theta)), dtype=itype)

    rows, cols = img.shape
    x,y = np.mgrid[:rows, :cols]

    for i, (cT, sT) in enumerate(zip(np.cos(theta), np.sin(theta))):
        rho = np.round_(cT * x[img] + sT * y[img]) - bins[0] + 1
        rho = rho.astype(itype)
        rho[(rho < 0) | (rho > nr_bins)] = 0
        bc = np.bincount(rho.flat)[1:]
        out[:len(bc), i] = bc

    return out, angles, bins

