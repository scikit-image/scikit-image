"""
Functions for calculating the "distance" between colors.

Implicit in these definitions of "distance" is the notion of "Just Noticible
Distance" (JND).  This represents the distance between colors where a human can
percieve different colors.  Humans are more sensitive to certain colors than
others, which different deltaE metrics correct for this with varying degrees of
sophistication.

The literature often mentions 1 as the minimum distance for visual
differentiation, but more recent studies (Mahy 1994) peg JND at 2.3

The delta-E notation comes from the German word for "Sensation" (Empfindung).

:author: Matt Terry

:license: modified BSD

Reference
---------
http://en.wikipedia.org/wiki/Color_difference

"""
from __future__ import division

import numpy as np

DEG = np.pi / 180


def _arctan2pi(b, a):
    """np.arctan2 mapped to (0, 2 * pi)"""
    ans = np.arctan2(b, a)
    ans += np.where(ans < 0, 2 * np.pi, 0.)
    return ans


def deltaE_cie76(lab1, lab2):
    """Euclidian distance between two points in in Lab color space

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparision color (Lab colorspace)

    Returns
    -------
    dE : array_like
        distance between colors `lab1` and `lab2`

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Color_difference
    .. [2] A. R. Robertson, "The CIE 1976 color-difference formulae,"
           Color Res. Appl. 2, 7-11 (1977).
    """
    l1, a1, b1 = np.rollaxis(lab1, -1)[:3]
    l2, a2, b2 = np.rollaxis(lab2, -1)[:3]
    return np.sqrt((l2 - l1) ** 2 + (a2 - a1) ** 2 + (b2 - b1) ** 2)


def deltaE_ciede94(lab1, lab2, kH=1, kC=1, kL=1, k1=0.045, k2=0.015):
    """Color difference according to CIEDE 94 standard

    Accomodates perceptual non-uniformites through the use of application
    specific scale factors (kH, kC, kL, k1, and k2).

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparision color (Lab colorspace)
    kH : float, optional
        Hue scale
    kC : float, optional
        Chroma scale
    kL : float, optional
        Lightness scale
    k1 : float, optional
        first scale parameter
    k2 : float, optional
        second scale parameter

    Returns
    -------
    dE : array_like
        color difference between `lab1` and `lab2`

    Notes
    -----
    deltaE_ciede94 is not symmetric with respect to lab1 and lab2.  CIEDE94
    defines the scales for the lightness, hue, and chroma in terms of the first
    color.  Consequently, the first color should be regarded as the "reference"
    color.

    kL, k1, k2 depend on the application and default to the values suggested
    for graphic arts

    Parameter   Graphic Arts    Textiles
    ----------  -------------   --------
    kL          1.000           2.000
    k1          0.045           0.048
    k2          0.015           0.014

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Color_difference
    .. [2] http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE94.html
    """
    l1, a1, b1 = np.rollaxis(lab1, -1)[:3]
    l2, a2, b2 = np.rollaxis(lab2, -1)[:3]

    dl = l1 - l2
    c1 = np.sqrt(a1 ** 2 + b1 ** 2)
    c2 = np.sqrt(a2 ** 2 + b2 ** 2)
    dc = c1 - c2
    dh_ab = np.sqrt(deltaE_cie76(lab1, lab2) ** 2 - dl ** 2 - dc ** 2)

    SL = 1
    SC = 1 + k1 * c1
    SH = 1 + k2 * c1

    ans = ((dl / (kL * SL)) ** 2 +
           (dc / (kC * SC)) ** 2 +
           (dh_ab / (kH * SH)) ** 2
           )
    return np.sqrt(ans)


def deltaE_ciede2000(lab1, lab2, kL=1, kC=1, kH=1):
    """Color difference as given by the CIEDE 2000 standard.

    CIEDE 2000 is a major revision of CIDE94.  The perceptual calibaration is
    largely based on experience with automotive paint on smooth surfaces.

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparision color (Lab colorspace)
    kL : float (range), optional
        pass
    kC : float (range), optional
        pass
    kH : float (range), optional
        pass

    Returns
    -------
    deltaE : array_like
        The distance between `lab1` and `lab2`

    Notes
    -----
    CIEDE 2000 assumes parametric weighting factors for the luminance, chroma,
    and hue (kL, kC, kH respectively).  These default to 1.

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Color_difference
    .. [2] http://www.ece.rochester.edu/~gsharma/ciede2000/ciede2000noteCRNA.pdf
           (doi:10.1364/AO.33.008069)
    .. [3] M. Melgosa, J. Quesada, and E. Hita, "Uniformity of some recent
           color metrics tested with an accurate color-difference tolerance
           dataset," Appl. Opt. 33, 8069-8077 (1994).
    """
    L1, a1, b1 = np.rollaxis(lab1, -1)[:3]
    L2, a2, b2 = np.rollaxis(lab2, -1)[:3]

    c1 = np.sqrt(a1 ** 2 + b1 ** 2)
    c2 = np.sqrt(a2 ** 2 + b2 ** 2)
    cbar = 0.5 * (c1 + c2)
    c7 = cbar ** 7
    G = 0.5 * (1 - np.sqrt(c7 / (c7 + 25 ** 7)))

    dL_prime = L2 - L1
    Lbar = 0.5 * (L1 + L2)

    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)

    c1_prime = np.sqrt(a1_prime ** 2 + b1 ** 2)
    c2_prime = np.sqrt(a2_prime ** 2 + b2 ** 2)
    cbar_prime = 0.5 * (c1_prime + c2_prime)
    dC_prime = c2_prime - c1_prime

    h1_prime = _arctan2pi(b1, a1_prime)
    h2_prime = _arctan2pi(b2, a2_prime)

    dh_prime = h2_prime - h1_prime

    cc = c1_prime * c2_prime
    mask1 = cc == 0.
    mask2 = np.logical_and(-mask1, dh_prime > np.pi)
    mask3 = np.logical_and(-mask1, dh_prime < -np.pi)
    dh_prime = np.where(mask1, 0., dh_prime)
    dh_prime += np.where(mask2, 2 * np.pi, 0)
    dh_prime -= np.where(mask3, 2 * np.pi, 0)

    dH_prime = 2 * np.sqrt(cc) * np.sin(dh_prime / 2)

    Hbar_prime = h1_prime + h2_prime
    mask0 = np.logical_and(np.abs(h1_prime - h2_prime) > np.pi, cc != 0.)
    mask1 = np.logical_and(mask0, Hbar_prime < 2 * np.pi)
    mask2 = np.logical_and(mask0, Hbar_prime >= 2 * np.pi)

    Hbar_prime += np.where(mask1, 2 * np.pi, 0)
    Hbar_prime -= np.where(mask2, 2 * np.pi, 0)
    Hbar_prime *= np.where(cc == 0., 2, 1)
    Hbar_prime *= 0.5

    T = (1 -
         0.17 * np.cos(Hbar_prime - 30 * DEG) +
         0.24 * np.cos(2 * Hbar_prime) +
         0.32 * np.cos(3 * Hbar_prime + 6 * DEG) -
         0.20 * np.cos(4 * Hbar_prime - 63 * DEG)
         )
    dTheta = 30 * DEG * np.exp(-((Hbar_prime / DEG - 275) / 25) ** 2)
    c7 = cbar_prime ** 7
    Rc = 2 * np.sqrt(c7 / (c7 + 25 ** 7))

    term = (Lbar - 50) ** 2
    SL = 1 + 0.015 * term / np.sqrt(20 + term)
    SC = 1 + 0.045 * cbar_prime
    SH = 1 + 0.015 * cbar_prime * T

    RT = -np.sin(2 * dTheta) * Rc

    l_term = dL_prime / (kL * SL)
    c_term = dC_prime / (kC * SC)
    h_term = dH_prime / (kH * SH)
    r_term = RT * c_term * h_term

    dE2 = l_term ** 2
    dE2 += c_term ** 2
    dE2 += h_term ** 2
    dE2 += r_term
    return np.sqrt(dE2)


def deltaE_cmc(lab1, lab2, kL=1, kC=1):
    """Color difference from the  CMC l:c standard.

    This color difference developed by the Colour Measurement Committee of the
    Socieity of Dyes and Colourists of Great Britian (CMC).  It is intended for
    use in the textile industry.

    The scale factors kL, kC set the weight given to differences in lightness
    and chroma relative to differences in hue.  The usual values are kL=2, kC=1
    for "acceptability" and kL=1, kC=1 for "imperceptability".  Colors with
    dE > 1 are "different" for the given scale factors.

    Parameters
    ----------
    lab1 : array_like
        reference color (Lab colorspace)
    lab2 : array_like
        comparision color (Lab colorspace)

    Returns
    -------
    dE : array_like
        distance between colors `lab1` and `lab2`

    Notes
    -----
    deltaE_cmc the defines the scales for the lightness, hue, and chroma
    in terms of the first color.  Consequently
    deltaE_cmc(lab1, lab2) != deltaE_cmc(lab2, lab1)

    References
    ----------
    .. [1] http://en.wikipedia.org/wiki/Color_difference
    .. [2] http://www.brucelindbloom.com/index.html?Eqn_DeltaE_CIE94.html
    .. [3] F. J. J. Clarke, R. McDonald, and B. Rigg, "Modification to the
           JPC79 colour-difference formula," J. Soc. Dyers Colour. 100, 128-132
           (1984).
    """
    l1, a1, b1 = np.rollaxis(lab1, -1)[:3]
    l2, a2, b2 = np.rollaxis(lab2, -1)[:3]

    c1 = np.sqrt(a1 ** 2 + b1 ** 2)
    c2 = np.sqrt(a2 ** 2 + b2 ** 2)
    dC = c1 - c2
    dl = l1 - l2
    dH = np.sqrt(deltaE_cie76(lab1, lab2) ** 2 - dl ** 2 - dC ** 2)

    dL = l1 - l2

    h1 = _arctan2pi(b1, a1)
    T = np.where(np.logical_and(h1 >= 164 * DEG, h1 <= 345 * DEG),
                 0.56 + 0.2 * np.abs(np.cos(h1 + 168 * DEG)),
                 0.36 + 0.4 * np.abs(np.cos(h1 + 35 * DEG))
                 )
    c1_4 = c1 ** 4
    F = np.sqrt(c1_4 / (c1_4 + 1900))

    SL = np.where(l1 < 16, 0.511, 0.040975 * l1 / (1. + 0.01765 * l1))
    SC = 0.638 + 0.0638 * c1 / (1. + 0.0131 * c1)
    SH = SC * (F * T + 1 - F)

    dE2 = (dL / (kL * SL)) ** 2
    dE2 += (dC / (kC * SC)) ** 2
    dE2 += (dH / SH) ** 2

    return np.sqrt(dE2)
