from __future__ import division
import numpy as np
from .._shared.utils import assert_nD
from ..util import img_as_float


def build_steerable(image, height=5, n_bands=4):
    """ Construct a Steerable subband decomposition of a gray scale image.
    
    The Steerable Pyramid is a linear multi-scale, multi-orientation image 
    decomposition that provides a useful front-end for image-processing 
    and computer vision applications. 

    Parameters
    ----------
    image : 2-D array
        Input image
    height : integer, optional
        Height of the steerable decomposition. This includes the counting of
        low pass and high pass subbands.
    n_bands : integer, optional
        Number of orientations in the Steerable decomposition.

    Returns
    -------
    coeff : list of lists of numpy array
        Subbands of Steerable decomposition,
        stored as a list of 'height' sublists,
        Sublists correspond to decreasing radius level in Steerable pyramid
        The first sublist contains the high pass subband.
        The last sublist contains the low pass subband.
        Intermediate sublists contain subbands from different orientations.

    References
    ----------
    .. [1] Simoncelli, E.P. & Freeman, W.T.
       (1995). The Steerable Pyramid: A Flexible Architecture for Multi-Scale
       Derivative Computation. In Proc. 2nd IEEE International Conf. on Image
       Proc., vol.III pp. 444-447, Oct 1995. 
       http://www.cns.nyu.edu/~eero/steerpyr/,
       http://www.cns.nyu.edu/pub/eero/simoncelli95b.pdf,
       DOI:10.1109/ICIP.1995.537667
    """

    assert_nD(image, 2)
    s = Steerable(height, n_bands)
    return s.build_scf_pyramid(image)


def recon_steerable(coeff):
    """ Reconstruct the image from its Steerable decomposition.

    The Steerable Pyramid is a linear multi-scale, multi-orientation image 
    decomposition that provides a useful front-end for image-processing 
    and computer vision applications. 

    Parameters
    ----------
    coeff : list of lists of numpy array
        Subbands of Steerable decomposition,
        stored as a list of 'height' sublists,
        Sublists correspond to decreasing radius level in Steerable pyramid
        The first sublist contains the high pass subband.
        The last sublist contains the low pass subband.
        Intermediate sublists contain subbands from different orientations.

    Returns
    -------
    image : 2-D array
        Reconstructed image from subbands of Steerable Pyramid decomposition.

    References
    ----------
    .. [1] E. P. Simoncelli and W. T. Freeman
        "The Steerable Pyramid: A Flexible Architecture
        for Multi-Scale Derivative Computation."
        http://www.cns.nyu.edu/~eero/steerpyr/
    """

    s = Steerable(height=len(coeff), n_bands=len(coeff[1]))
    return s.recon_scf_pyramid(coeff)


class Steerable:
    """Steerable Pyramid: a translation and rotation invariant free wavelet.
    
    The Steerable Pyramid is a linear multi-scale, multi-orientation 
    image decomposition that provides a useful front-end for image-processing 
    and computer vision applications. 

    """

    def __init__(self, height=5, n_bands=4):
        """
        Parameters
        ----------
        height : int
            Height of Steerable decomposition
            (including high pass and low pass)
        n_bands : int
            Number of orientations in Steerable decomposition
        """

        if not (height % 1 == 0 and n_bands % 1 == 0):
            raise ValueError("Height and n_bands must be integers")

        if height < 3:
            raise ValueError("Height must be at least 3")

        self.n_bands = int(n_bands)
        self.height = int(height)

    def build_scf_pyramid(self, im):
        """ Calculate Steerable subband decomposition of image 'im'
        
        Parameters
        ----------
        im : 2-D array
            Input gray scale image
        """
        assert_nD(im, 2)

        im = img_as_float(im)
        M, N = im.shape

        if min(M, N) < 2**self.height:
            raise ValueError("For a {} level decomposition,"
                             " all image dimensions must be at least"
                             " size {}.".format(self.height, 2**self.height))

        log_rad, angle = _logradius_angle_grid(M, N)
        Xrcos, Yrcos = _rcos_curve(1, -0.5)
        YIrcos = np.sqrt(1 - Yrcos)
        Yrcos = np.sqrt(Yrcos)

        lo0mask = _point_op(log_rad, YIrcos, Xrcos)
        hi0mask = _point_op(log_rad, Yrcos, Xrcos)

        imdft = np.fft.fftshift(np.fft.fft2(im))
        lo0dft = imdft * lo0mask

        coeff = self._build_pyr_level(
            lo0dft, log_rad, angle, Xrcos, Yrcos, self.height - 1)

        hi0dft = imdft * hi0mask
        hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft))

        coeff.insert(0, [hi0.real])

        return coeff

    def _build_pyr_level(self, lodft, log_rad, angle, Xrcos, Yrcos, ht):
        """ Recursively calculate Steerable subbands

        Parameters
        ----------
        lodft : 2-D array
            DFT matrix of the higher layer
        log_rad, angle: 2-D array
            Helper matrices to help create the DFT mask
        Xrcos, Yrcos: 2-D array
            Represents the desired filter
        ht: int
            Current level of the pyramid that are being built

        Notes
        -----
        This method is called recursively until ht reaches 1.
        """
        if (ht <= 1):
            lo0 = np.fft.ifft2(np.fft.ifftshift(lodft))
            coeff = [[lo0.real]]

        else:
            Xrcos = Xrcos - 1

            # ==================== Orientation bandpass =======================
            himask = _point_op(log_rad, Yrcos, Xrcos)

            lutsize = 1024
            Xcosn = np.pi * \
                np.array(range(-(2 * lutsize + 1), (lutsize + 2))) / lutsize
            order = self.n_bands - 1
            const = np.power(2, 2 * order) * \
                np.square(np.math.factorial(order)) / \
                (self.n_bands * np.math.factorial(2 * order))

            alpha = (Xcosn + np.pi) % (2 * np.pi) - np.pi
            Ycosn = 2 * np.sqrt(const) * \
                np.power(np.cos(Xcosn), order) * (np.abs(alpha) < np.pi / 2)

            orients = []

            for b in range(self.n_bands):
                anglemask = _point_op(
                    angle, Ycosn, Xcosn + np.pi * b / self.n_bands)
                banddft = np.power(np.complex(
                    0, -1), self.n_bands - 1) * lodft * anglemask * himask
                band = np.fft.ifft2(np.fft.ifftshift(banddft))
                orients.append(band)

            # ================== Subsample lowpass ============================
            dims = np.array(lodft.shape)

            lostart = np.ceil((dims + 0.5) / 2) - \
                np.ceil((np.ceil((dims - 0.5) / 2) + 0.5) / 2)
            loend = lostart + np.ceil((dims - 0.5) / 2)

            lostart = lostart.astype(np.int32)
            loend = loend.astype(np.int32)

            log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]

            YIrcos = np.abs(np.sqrt(1 - Yrcos * Yrcos))
            lomask = _point_op(log_rad, YIrcos, Xrcos)

            lodft = lomask * lodft

            coeff = self._build_pyr_level(
                lodft, log_rad, angle, Xrcos, Yrcos, ht - 1)
            coeff.insert(0, orients)

        return coeff

    def _recon_pyr_level(self, coeff, log_rad, Xrcos, Yrcos, angle):
        """
        Parameters
        ----------
        lodft : 2-D array
            DFT matrix of the higher layer
        log_rad, angle: 2-D array
            Helper matrices to help create the DFT mask
        Xrcos, Yrcos: 2-D array
            Represents the desired filter
        ht: int
            Current level of the pyramid that are being built

        Notes
        -----
        This method is called recursively until ht reaches 1.
        """

        if (len(coeff) == 1):
            return np.fft.fftshift(np.fft.fft2(coeff[0][0]))

        else:

            Xrcos = Xrcos - 1

            # ========================== Orientation residue===================
            himask = _point_op(log_rad, Yrcos, Xrcos)

            lutsize = 1024
            Xcosn = np.pi * \
                np.array(range(-(2 * lutsize + 1), (lutsize + 2))) / lutsize
            order = self.n_bands - 1
            const = np.power(2, 2 * order) * \
                np.square(np.math.factorial(order)) / \
                (self.n_bands * np.math.factorial(2 * order))

            Ycosn = np.sqrt(const) * np.power(np.cos(Xcosn), order)

            orientdft = np.zeros(coeff[0][0].shape)

            for b in range(self.n_bands):
                anglemask = _point_op(
                    angle, Ycosn, Xcosn + np.pi * b / self.n_bands)
                banddft = np.fft.fftshift(np.fft.fft2(coeff[0][b]))
                orientdft = orientdft + \
                    np.power(np.complex(0, 1), order) * \
                    banddft * anglemask * himask

            # ============== Lowpass component are upsampled and convoluted ===
            dims = np.array(coeff[0][0].shape)

            lostart = (np.ceil((dims + 0.5) / 2) -
                       np.ceil((np.ceil((dims - 0.5) / 2) + 0.5) / 2))\
                .astype(np.int32)

            loend = lostart + np.ceil((dims - 0.5) / 2).astype(np.int32)

            nlog_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            nangle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            YIrcos = np.sqrt(np.abs(1 - Yrcos * Yrcos))
            lomask = _point_op(nlog_rad, YIrcos, Xrcos)

            nresdft = self._recon_pyr_level(
                coeff[1:], nlog_rad, Xrcos, Yrcos, nangle)

            resdft = np.zeros(dims, 'complex')
            resdft[lostart[0]:loend[0], lostart[1]:loend[1]] = nresdft * lomask

            return resdft + orientdft

    def recon_scf_pyramid(self, coeff):
        """ Reconstruct the image from its Steerable decomposition.

        Parameters
        ----------
        coeff : list of lists of numpy array
            Subbands of Steerable decomposition,
            stored as a list of 'height' sublists,
            Sublists correspond to decreasing radius level in Steerable pyramid
            The first sublist contains the high pass subband.
            The last sublist contains the low pass subband.
            Intermediate sublists contain subbands from different orientations.

        Returns
        -------
        image : 2-D array
            Reconstructed image from subbands of Steerable Pyramid decomposition
        """

        if (self.height != len(coeff)):
            raise ValueError("Height of coeff should be %d" % self.height)

        if (len(coeff[0]) != 1):
            raise ValueError("High pass sublist should be of length one")

        if (len(coeff[-1]) != 1):
            raise ValueError("Low pass sublist should be of length one")

        for i in range(1, self.height - 1):
            if (self.n_bands != len(coeff[i])):
                raise ValueError(
                    "Size of intermediate sublist should be %d" % self.n_bands)

        r, c = coeff[0][0].shape
        log_rad, angle = _logradius_angle_grid(r, c)

        Xrcos, Yrcos = _rcos_curve(1, -0.5)
        Yrcos = np.sqrt(Yrcos)
        YIrcos = np.sqrt(np.abs(1 - Yrcos * Yrcos))

        lo0mask = _point_op(log_rad, YIrcos, Xrcos)
        hi0mask = _point_op(log_rad, Yrcos, Xrcos)

        tempdft = self._recon_pyr_level(
            coeff[1:], log_rad, Xrcos, Yrcos, angle)

        hidft = np.fft.fftshift(np.fft.fft2(coeff[0][0]))
        outdft = tempdft * lo0mask + hidft * hi0mask

        return np.fft.ifft2(np.fft.ifftshift(outdft)).real


def _logradius_angle_grid(m, n):
    """
    Helper function that create a grid containing radius from center and angle.

    Parameters
    ----------
    m, n : int
        Size of the desired grid

    Returns
    -------
    log_rad : [mxn] array
        Log of the radius with respect to the center of the matrix
    angle : [mxn] array
        The angle of the line originated from center of the matrix
    """

    ctrm = np.ceil((m + 0.5) / 2).astype(int)
    ctrn = np.ceil((n + 0.5) / 2).astype(int)
    xv, yv = np.meshgrid((np.array(range(n)) + 1 - ctrn) / (n / 2),
                         (np.array(range(m)) + 1 - ctrm) / (m / 2),
                         sparse=True)

    rad = np.sqrt(xv**2 + yv**2)
    rad[ctrm - 1, ctrn - 1] = rad[ctrm - 1, ctrn - 2]
    log_rad = np.log2(rad)
    angle = np.arctan2(yv, xv)

    return log_rad, angle


def _rcos_curve(width, position):
    """
    Raised cosine 1D curve.

    Parameters
    ----------
    width: float
        width of the transition
    position: float
        position of the cut

    Returns
    -------
    X, Y: define a 1D raised cosine curve

    """
    N = 256
    X = np.pi * np.array(range(-N - 1, 2)) / 2 / N

    Y = np.cos(X)**2
    Y[0] = Y[1]
    Y[N + 2] = Y[N + 1]

    X = position + 2 * width / np.pi * (X + np.pi / 4)
    return X, Y


def _point_op(mask, Y, X):
    """
    Given a 1D curve defined by X, Y, convert it to a 2D mask.

    Parameters
    ----------
    mask: 2-D array

    X, Y: vector
        define a 1D raised cosine curve

    Returns
    -------
    out: 2-D array
        2-D raised cosine circular mask

    """
    out = np.interp(mask.ravel(), X, Y)
    return np.reshape(out, mask.shape)
