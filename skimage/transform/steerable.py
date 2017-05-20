from __future__ import division
import numpy as np
import scipy.misc as sc
from .._shared.utils import assert_nD
from ..util import img_as_float


def build_steerable(image, height=5, sampling=True):
    """
    Parameters
    ----------
    image : 2-D array
        image to process
    height : integer, optional
        Height of the steerable decomposition. This includes the counting of
        low pass and high pass.
    sampling : booelan, optional
        If sampling is true, then use subsampling when generating the subbands.
        If sampling is false, only filters are applied.


    Returns
    -------
    coeff : is a list containing subbands, which has the following structure
            first element is high pass subband
            last element is low pass subband
            the elements in between are a list  containing subbands from different orientations

    References
    ----------
    """
    assert_nD(image, 2)

    s = Steerable(height)
    return s.build_scf_pyramid(image)


def recon_steerable(coeff):
    """
    Parameters
    ----------
    coeff : is a list containing subbands, which has the following structure
            first element is high pass subband
            last element is low pass subband
            the elements in between are a list  containing subbands from different orientations

    Returns
    -------
    image : 2-D array
        reconstructed image from subbands of Steerable Pyramid decomposition

    References
    ----------
    """
    height = len(coeff)
    s = Steerable(height)
    return s.recon_scf_pyramid(coeff)


class Steerable:
    """Steerable Pyramid: a translation and rotation invariant free wavelet
    """

    def __init__(self, height=5):
        """
        Parameters
        ----------
        height : height of the Steerable Decomposition (including high pass and low pass)
        """
        self.nbands = 4
        self.height = height
        self.isSample = True

    def build_scf_pyramid(self, im):
        """
        Parameters
        ----------
        im : 2-D array
        """
        assert_nD(im, 2)

        im = img_as_float(im)

        M, N = im.shape
        log_rad, angle = self.base(M, N)
        Xrcos, Yrcos = self.rcos_curve(1, -0.5)
        Yrcos = np.sqrt(Yrcos)
        YIrcos = np.sqrt(1 - Yrcos * Yrcos)

        lo0mask = self.point_op(log_rad, YIrcos, Xrcos)
        hi0mask = self.point_op(log_rad, Yrcos, Xrcos)

        imdft = np.fft.fftshift(np.fft.fft2(im))
        lo0dft = imdft * lo0mask

        coeff = self.build_pyr_level(
            lo0dft, log_rad, angle, Xrcos, Yrcos, self.height - 1)

        hi0dft = imdft * hi0mask
        hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft))

        coeff.insert(0, hi0.real)

        return coeff

    def build_pyr_level(self, lodft, log_rad, angle, Xrcos, Yrcos, ht):
        """
        Parameters
        ----------
        lodft : DFT matrix of the higher layer
        log_rad, angle: helper to help create the DFT mask
        Xrcos, Yrcos: represents the desired filter
        ht: current level of the pyramid that are being built

        Notes
        -----
        This method is called recursively until ht reaches 1.
        """
        if (ht <= 1):
            lo0 = np.fft.ifft2(np.fft.ifftshift(lodft))
            coeff = [lo0.real]

        else:
            Xrcos = Xrcos - 1

            # ==================== Orientation bandpass =======================
            himask = self.point_op(log_rad, Yrcos, Xrcos)

            lutsize = 1024
            Xcosn = np.pi * \
                np.array(range(-(2 * lutsize + 1), (lutsize + 2))) / lutsize
            order = self.nbands - 1
            const = np.power(2, 2 * order) * np.square(sc.factorial(order)
                                                       ) / (self.nbands * sc.factorial(2 * order))

            alpha = (Xcosn + np.pi) % (2 * np.pi) - np.pi
            Ycosn = 2 * np.sqrt(const) * np.power(np.cos(Xcosn),
                                                  order) * (np.abs(alpha) < np.pi / 2)

            orients = []

            for b in range(self.nbands):
                anglemask = self.point_op(
                    angle, Ycosn, Xcosn + np.pi * b / self.nbands)
                banddft = np.power(np.complex(
                    0, -1), self.nbands - 1) * lodft * anglemask * himask
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
            lomask = self.point_op(log_rad, YIrcos, Xrcos)

            lodft = lomask * lodft

            coeff = self.build_pyr_level(
                lodft, log_rad, angle, Xrcos, Yrcos, ht - 1)
            coeff.insert(0, orients)

        return coeff

    def recon_pyr_level(self, coeff, log_rad, Xrcos, Yrcos, angle):
        """
        Parameters
        ----------
        lodft : DFT matrix of the higher layer
        log_rad, angle: helper to help create the DFT mask
        Xrcos, Yrcos: represents the desired filter
        ht: current level of the pyramid that are being built

        Notes
        -----
        This method is called recursively until ht reaches 1.
        """

        if (len(coeff) == 1):
            return np.fft.fftshift(np.fft.fft2(coeff[0]))

        else:

            Xrcos = Xrcos - 1

            # ========================== Orientation residue===================
            himask = self.point_op(log_rad, Yrcos, Xrcos)

            lutsize = 1024
            Xcosn = np.pi * \
                np.array(range(-(2 * lutsize + 1), (lutsize + 2))) / lutsize
            order = self.nbands - 1
            const = np.power(2, 2 * order) * np.square(sc.factorial(order)
                                                       ) / (self.nbands * sc.factorial(2 * order))
            Ycosn = np.sqrt(const) * np.power(np.cos(Xcosn), order)

            orientdft = np.zeros(coeff[0][0].shape)

            for b in range(self.nbands):
                anglemask = self.point_op(
                    angle, Ycosn, Xcosn + np.pi * b / self.nbands)
                banddft = np.fft.fftshift(np.fft.fft2(coeff[0][b]))
                orientdft = orientdft + \
                    np.power(np.complex(0, 1), order) * \
                    banddft * anglemask * himask

            # ============== Lowpass component are upsampled and convoluted ===
            dims = np.array(coeff[0][0].shape)

            lostart = (np.ceil((dims + 0.5) / 2) -
                       np.ceil((np.ceil((dims - 0.5) / 2) + 0.5) / 2)).astype(np.int32)
            loend = lostart + np.ceil((dims - 0.5) / 2).astype(np.int32)

            nlog_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            nangle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            YIrcos = np.sqrt(np.abs(1 - Yrcos * Yrcos))
            lomask = self.point_op(nlog_rad, YIrcos, Xrcos)

            nresdft = self.recon_pyr_level(
                coeff[1:], nlog_rad, Xrcos, Yrcos, nangle)

            res = np.fft.fftshift(np.fft.fft2(nresdft))

            resdft = np.zeros(dims, 'complex')
            resdft[lostart[0]:loend[0], lostart[1]:loend[1]] = nresdft * lomask

            return resdft + orientdft

    def recon_scf_pyramid(self, coeff):
        """
        Parameters
        ----------
        coeff : is a list containing subbands, which has the following structure
        first element is high pass subband
        last element is low pass subband
        the elements in between are a list  containing subbands from different orientations

        """
        if (self.nbands != len(coeff[1])):
            raise ValueError(
                "The number of orientations of subband and steerable pyramid does not match")

        M, N = coeff[0].shape
        log_rad, angle = self.base(M, N)

        Xrcos, Yrcos = self.rcos_curve(1, -0.5)
        Yrcos = np.sqrt(Yrcos)
        YIrcos = np.sqrt(np.abs(1 - Yrcos * Yrcos))

        lo0mask = self.point_op(log_rad, YIrcos, Xrcos)
        hi0mask = self.point_op(log_rad, Yrcos, Xrcos)

        tempdft = self.recon_pyr_level(coeff[1:], log_rad, Xrcos, Yrcos, angle)

        hidft = np.fft.fftshift(np.fft.fft2(coeff[0]))
        outdft = tempdft * lo0mask + hidft * hi0mask

        return np.fft.ifft2(np.fft.ifftshift(outdft)).real.astype(np.int32)

    def base(self, m, n):
        """
        Parameters
        ----------
        m, n : size of the desired

        Returns
        -------
        log_rad : [mxn] array: log of the radius to the center of the matrix
        angle : [mxn] array: the angle of the line start from the center

        Notes
        -----
        Helper function that create a grid containing radius from center and angle
        """

        ctrm = np.ceil((m + 0.5) / 2).astype(int)
        ctrn = np.ceil((n + 0.5) / 2).astype(int)
        xv, yv = np.meshgrid((np.array(range(n)) + 1 - ctrn) / (n / 2),
                             (np.array(range(m)) + 1 - ctrm) / (m / 2))

        rad = np.sqrt(xv**2 + yv**2)
        rad[ctrm - 1, ctrn - 1] = rad[ctrm - 1, ctrn - 2]
        log_rad = np.log2(rad)

        angle = np.arctan2(yv, xv)

        return log_rad, angle

    def rcos_curve(self, width, position):
        """
        Parameters
        ----------
        width: width of the transition
        position: where the cut is

        Returns
        -------
        X, Y: define a 1D curve, that look similar to a tanh

        Notes
        -----
        Calculate the curve of the filter
        """
        N = 256
        X = np.pi * np.array(range(-N - 1, 2)) / 2 / N

        Y = np.cos(X)**2
        Y[0] = Y[1]
        Y[N + 2] = Y[N + 1]

        X = position + 2 * width / np.pi * (X + np.pi / 4)
        return X, Y

    def point_op(self, im, Y, X):
        """
        Notes
        -----
        Given a 1D curve defined by X, Y, convert it to a 2D mask
        """
        out = np.interp(im.flatten(), X, Y)
        return np.reshape(out, im.shape)
