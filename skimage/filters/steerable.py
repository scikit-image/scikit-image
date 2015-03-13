from __future__ import division
import numpy as np
import scipy.misc as sc
from .._shared.utils import assert_nD

def buildSteerable(image, height = 5):
    assert (len(image.shape) == 2)

    if sampling:
        s = Steerable(height)
    else:
        s = SteerableNoSub(height)
    return s.buildSCFpyr(image)

def reconSteerable(coeff, height = 5):
    if sampling:
        s = Steerable(height)
    else:
        s = SteerableNoSub(height)

    return s.reconSCFpyr(coeff)

class Steerable:
    def __init__(self, height = 5):
        """
        height is the total height, including highpass and lowpass
        """
        self.nbands = 4
        self.height = height
        self.isSample = True

    def buildSCFpyr(self, im):
        assert len(im.shape) == 2, 'Input image must be grayscale'

        M, N = im.shape
        log_rad, angle = self.base(M, N)
        Xrcos, Yrcos = self.rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)
        YIrcos = np.sqrt(1 - Yrcos*Yrcos)

        lo0mask = self.pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = self.pointOp(log_rad, Yrcos, Xrcos)

        imdft = np.fft.fftshift(np.fft.fft2(im))
        lo0dft = imdft * lo0mask

        coeff = self.buildSCFpyrlevs(lo0dft, log_rad, angle, Xrcos, Yrcos, self.height - 1)

        hi0dft = imdft * hi0mask
        hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft))

        coeff.insert(0, hi0.real)

        return coeff

    def getlist(self, coeff):
        straight = [bands for scale in coeff[1:-1] for bands in scale]
        straight = [coeff[0]] + straight + [coeff[-1]]
        return straight

    def buildSCFpyrlevs(self, lodft, log_rad, angle, Xrcos, Yrcos, ht):
        if (ht <=1):
            lo0 = np.fft.ifft2(np.fft.ifftshift(lodft))
            coeff = [lo0.real]
        
        else:
            Xrcos = Xrcos - 1

            # ==================== Orientation bandpass =======================
            himask = self.pointOp(log_rad, Yrcos, Xrcos)

            lutsize = 1024
            Xcosn = np.pi * np.array(range(-(2*lutsize+1),(lutsize+2)))/lutsize
            order = self.nbands - 1
            const = np.power(2, 2*order) * np.square(sc.factorial(order)) / (self.nbands * sc.factorial(2*order))

            alpha = (Xcosn + np.pi) % (2*np.pi) - np.pi
            Ycosn = 2*np.sqrt(const) * np.power(np.cos(Xcosn), order) * (np.abs(alpha) < np.pi/2)

            orients = []

            for b in range(self.nbands):
                anglemask = self.pointOp(angle, Ycosn, Xcosn + np.pi*b/self.nbands)
                banddft = np.power(np.complex(0,-1), self.nbands - 1) * lodft * anglemask * himask
                band = np.fft.ifft2(np.fft.ifftshift(banddft))
                orients.append(band.real)

            # ================== Subsample lowpass ============================
            dims = np.array(lodft.shape)
            
            lostart = np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2)
            loend = lostart + np.ceil((dims-0.5)/2)

            lostart = lostart.astype(int)
            loend = loend.astype(int)

            log_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            angle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            lodft = lodft[lostart[0]:loend[0], lostart[1]:loend[1]]
            YIrcos = np.abs(np.sqrt(1 - Yrcos*Yrcos))
            lomask = self.pointOp(log_rad, YIrcos, Xrcos)

            lodft = lomask * lodft

            coeff = self.buildSCFpyrlevs(lodft, log_rad, angle, Xrcos, Yrcos, ht-1)
            coeff.insert(0, orients)

        return coeff

    def reconSCFpyrLevs(self, coeff, log_rad, Xrcos, Yrcos, angle):

        if (len(coeff) == 1):
            return np.fft.fftshift(np.fft.fft2(coeff[0]))

        else:

            Xrcos = Xrcos - 1
            
            # ========================== Orientation residue==========================
            himask = self.pointOp(log_rad, Yrcos, Xrcos)

            lutsize = 1024
            Xcosn = np.pi * np.array(range(-(2*lutsize+1),(lutsize+2)))/lutsize
            order = self.nbands - 1
            const = np.power(2, 2*order) * np.square(sc.factorial(order)) / (self.nbands * sc.factorial(2*order))
            Ycosn = np.sqrt(const) * np.power(np.cos(Xcosn), order)

            orientdft = np.zeros(coeff[0][0].shape)

            for b in range(self.nbands):
                anglemask = self.pointOp(angle, Ycosn, Xcosn + np.pi* b/self.nbands)
                banddft = np.fft.fftshift(np.fft.fft2(coeff[0][b]))
                orientdft = orientdft + np.power(np.complex(0,1), order) * banddft * anglemask * himask

            # ============== Lowpass component are upsampled and convoluted ============
            dims = np.array(coeff[0][0].shape)
            
            lostart = np.ceil((dims+0.5)/2) - np.ceil((np.ceil((dims-0.5)/2)+0.5)/2) 
            loend = lostart + np.ceil((dims-0.5)/2) 

            lostart = lostart.astype(int)
            loend = loend.astype(int)

            nlog_rad = log_rad[lostart[0]:loend[0], lostart[1]:loend[1]]
            nangle = angle[lostart[0]:loend[0], lostart[1]:loend[1]]
            YIrcos = np.sqrt(np.abs(1 - Yrcos * Yrcos))
            lomask = self.pointOp(nlog_rad, YIrcos, Xrcos)

            nresdft = self.reconSCFpyrLevs(coeff[1:], nlog_rad, Xrcos, Yrcos, nangle)

            res = np.fft.fftshift(np.fft.fft2(nresdft))

            resdft = np.zeros(dims, 'complex')
            resdft[lostart[0]:loend[0], lostart[1]:loend[1]] = nresdft * lomask

            return resdft + orientdft

    def reconSCFpyr(self, coeff):

        if (self.nbands != len(coeff[1])):
            raise Exception("Unmatched number of orientations")

        M, N = coeff[0].shape
        log_rad, angle = self.base(M, N)

        Xrcos, Yrcos = self.rcosFn(1, -0.5)
        Yrcos = np.sqrt(Yrcos)
        YIrcos = np.sqrt(np.abs(1 - Yrcos*Yrcos))

        lo0mask = self.pointOp(log_rad, YIrcos, Xrcos)
        hi0mask = self.pointOp(log_rad, Yrcos, Xrcos)

        tempdft = self.reconSCFpyrLevs(coeff[1:], log_rad, Xrcos, Yrcos, angle)

        hidft = np.fft.fftshift(np.fft.fft2(coeff[0]))
        outdft = tempdft * lo0mask + hidft * hi0mask

        return np.fft.ifft2(np.fft.ifftshift(outdft)).real.astype(int)


    def base(self, m, n):
        
        x = np.linspace(-(m // 2)/(m / 2), (m // 2)/(m / 2) - (1 - m % 2)*2/m , num = m)
        y = np.linspace(-(n // 2)/(n / 2), (n // 2)/(n / 2) - (1 - n % 2)*2/n , num = n)

        xv, yv = np.meshgrid(y, x)

        angle = np.arctan2(yv, xv)

        rad = np.sqrt(xv**2 + yv**2)
        rad[m//2][n//2] = rad[m//2][n//2 - 1]
        log_rad = np.log2(rad)

        return log_rad, angle

    def rcosFn(self, width, position):
        N = 256
        X = np.pi * np.array(range(-N-1, 2))/2/N

        Y = np.cos(X)**2
        Y[0] = Y[1]
        Y[N+2] = Y[N+1]

        X = position + 2*width/np.pi*(X + np.pi/4)
        return X, Y

    def pointOp(self, im, Y, X):
        out = np.interp(im.flatten(), X, Y)
        return np.reshape(out, im.shape)