""" Scattering transform code licensed under
    BSD license.
    Original author: Sira Ferradans, based on a code developed by
    Ivan Dokmanic and Michael Eickenberg
"""
import numpy as np
from skimage._shared.utils import assert_nD
from skimage.filters.filter_bank import multiresolution_filter_bank_morlet2d
import warnings, sys


def _apply_fourier_mult(signals, filters):
    """Pointwise multiplication of the filters and signals 
    in the Fourier domain.
        This function applies a filtering of the signals in the
    Fourier domain using the filters, also in the Fourier domain. Note
    that we apply for every signal (num_signal) a number L of filters, 
    thus the output is (num_signals, L, N, N).
        Parameters
        ----------
        signals: ndarray (3D)
            Signals in the Fourier domain stacked as (num_signals, N, N).
            Note that color images can be stacked as 'num_signals'.
        filters: ndarray (3D)
            Filters in the Fourier domain, stacked as (L, N, N) where
            L is the num. of filters to apply
        Returns
        -------
        filtered_signals : ndarray
            Filtered signals in the Fourier domain,
            stacked as (num_signals,L,N,N)
        """
    filtered_s = signals[:, np.newaxis, :, :] * filters[np.newaxis, :, :, :]
    return filtered_s


def _subsample(X, j):
    """Spatial subsampling on the last two dimensions of X at a rate that
    depends on 2**j.
    Parameters
    ----------
    X : array like variable.
        3D ndarray with shape (N, L, n, n) or (N, n, n). The subsampling is 
        produced in the last two dimensions (n, n)
    j : int
        Rate of subsampling is 2**j
    Returns
    -------
    XX : ndarray
        Subsampled images
    """
    dsf = 2 ** j
    return dsf * X[..., ::dsf, ::dsf]


def _apply_lowpass(img, phi, J, n_scat):
    """Apply a low pass filter 'phi' to images 'img' and subsample.
    Convolution the filter phi (in the Fourier domain) on the set of images
    defined by img. The convolution is done in the Fourier domain, thus it is
    a pointwise multiplication of the filter phi and the images, stored in the
    last 2 dimensions of img. Then the images are subsampled at the appropiate
    rate to have 'n_scat' spatial coefficients
    Parameters
    ----------
    img : ndarray of (N, L, n, n) or (N, n, n) shape.
        Input images in the spatial domain, stacked along the first dimensions.
    phi : Low pass filter as a 2D ndarray
        Low pass filter in the Fourier domain, stored as (n, n) matrix
    J : int
        Rate of subsampling 2**J
    n_scat : int
        number of spatial coefficients of the scattering vector
    Returns
    -------
    XX : ndarray
        stacked images after being filtered and subsampled
    """
    img_filtered = np.real(np.fft.ifft2(np.fft.fft2(img) * phi))
    n_spatial = img.shape[-1]
    ds = int(n_spatial / n_scat)
    return 2 ** (J - 1) * img_filtered[..., ::ds, ::ds]


def scattering(x, wavelet_filters=None, m=2):
    """Compute the scattering transform of a signal (or set of signals).
    Given 'x', a set of 2D signals, this function computes the
    scattering transform of these signals using the filter bank
    'wavelet_filters'.
    Notes
    -----
    **Boundary values: The scattering transform applies a set of convolutions
    to the input signals. These convolutions are computed as the point-wise
    multiplication in the Fourier domain, thus the boundary values of the
    image are circular or cyclic. In case other
    kind of boundary values are needed, for instance zero-padded, the images
    should be preprocessed appropriately before calling this function.
    **Shape of x: The signals x must be squared shaped, thus (N, px, px)
    and not (N, px, py) for py != px. In case the images are rectangular,
    they will be cropped to the smallest dimension, px or py.
    Parameters
    ----------
    x : array_like
        3D dnarray with N images (2D arrays) of size (px, px), thus x has size
         (N, px, px)
        In case the array is rectangular (N, px, py) for px not equal to py, t
        he images will be cropped.
    wavelet_filters : Dictionary with the multiresolution wavelet filter bank
        Dictionary of vectors obtained after calling:
            >>>> px = 32 #number of pixels of the images
            >>>> J = np.log2(px) #number of scales
            >>>> wavelet_filters = multiresolution_filter_bank_morlet2d(px, J=J)
    m : int
        Order of the scattering transform, which can be 0, 1 or 2.
    Returns
    -------
    S : 4D array_like
        Scattering transform of the x signals, of size
        (N, num_coeffs, spatial_coefs, spatial_coefs). For more information
        see _[1] _[2]
    U : array_like
        Result before applying the lowpass filter and subsampling.
    S_tree : dictionary
        Dictionary that allows to access the scattering coefficients (S)
        according to the layer and indices. More specifically:
            Zero-order layer: The only available key is 0:
            S_tree[0] : returns all the coefficients of the 0-order scattering
             transform

            First-order layer: The keys are tuples with (j, l) indexing
            S_tree[(j, l)] : returns the first order coefficients for scale 'j'
             and angle 'l'
            S_tree[((j1, l1), (j2, l2))] : the second order for scale 'j1',
            angle 'l1' on the first layer, and 'j2', 'l2' in the second layer.

        The number of coefficients for each entry is
        (N, spatial_coefs, spatial_coefs)
    Raises
    ------
    UserWarning
        If the size of x is not (N, px, px) and informs that the images with be
        cropped to (N, px, px)
    UserWarning
        If no wavelet filters, the function creates a multiresolution set of
        filters with predefined settings, but warns about the parameters and
        suggests precomputing the filters.
    ValueError
        If the value of m is not 0, 1, or 2.
    References
    ----------
    .. [1] Bruna, J., Mallat, S. 'Invariant Scattering Convolutional Networks'.
    IEEE Transactions on PAMI, 2012.
    .. [2] Oyallon, E. et Mallat, S. 'Deep Roto-translation Scattering for
    Object Classification'. CVPR 2015.
    Examples
    --------
    Processing a set of 3 (randomly generated) images:
       >>>> import numpy as np
       >>>> from scattering.filter_bank \
                import multiresolution_filter_bank_morlet2d
       >>>> from scattering.scattering import scattering
       >>>> px = 32 #number of pixels of the images
       >>>> images = range(0,3*px*px).reshape(3,px,px) #create images
       >>>> wavelet_filters,lp = \
                multiresolution_filter_bank_morlet2d(px,J=np.log2(px))
       >>>> S,U = scattering(images,m=2)
    """
    # Check that the input data is correct:
    # 1.- Signals are squared, otherwise, crop
    assert_nD(x, 3, 'x')  # check that the images are correctly stacked
    num_signals, px, py = x.shape
    if px != py:
        warning_string = "Variable x has shape {0}, " \
                         "which is not in format (N,px,px). " \
                         "We crop to the smallest dimension."
        warnings.warn(warning_string.format(x.shape))
        px = min(px, py)
        x = x[:, 0:px, 0:px]
    # 2.- If we dont have filters, get them with the default values, J=3, L=8
    if wavelet_filters is None:
        J = int(min(np.log2(px), 3))
        L = 8
        warning_string = "No filter input, we create a " \
                         "Morlet filter bank with " \
                         "J= {0} and L={1}. " \
                         "Strongly suggest creating " \
                         "the filters before hand and" \
                         " pass them as a parameter."
        warnings.warn(warning_string.format(J, L))
        wavelet_filters, lp = multiresolution_filter_bank_morlet2d(px,
                                                                   J=J, L=L)
    else:
        J = len(wavelet_filters['psi'][0])  # number of scales
        L = len(wavelet_filters['psi'][0][0])  # number of orientations
    num_signals = x.shape[0]
    # 3.- Check the Order (m) of the scattering transform, can only be 0,1,2,
    # and that gives us different number of scattering coefficients

    if m == 0:
        num_coefs = 1
    elif m == 1:
        num_coefs = 1 + J * L
    elif m == 2:
        num_coefs = int(1 + J * L + (L ** 2) * (J - 1) * J / 2)
    else:
        error_string = "Parameter m out of bounds, " \
                       "valid values are 0,1,2 not {0}"
        raise ValueError(error_string.format(m))
        return
    # constants
    spatial_coefs = int(x.shape[1] / 2 ** (J - 1))
    # subsample at a rate a bit lower than the critical frequency
    oversample = 1
    U = []
    V = []
    v_resolution = []
    current_res = 0
    # vars to return (where we save the scattering and its accessing
    # tree-structure (a dictionary)
    S = np.ndarray((num_signals, num_coefs, spatial_coefs, spatial_coefs))
    # allows access to the coefficients (S) using the tree structure
    S_tree = {}
    # Start computing the scattering coefficients
    # Zero order scattering coeffs
    S[:, 0, :, :] = _apply_lowpass(x, wavelet_filters['phi'][current_res],
                                   J,  spatial_coefs)
    S_tree[0] = S[:, 0, :, :]
    l_indexing = range(0, L)
    # First order scattering coeffs
    if m > 0:
        Sm1 = S[:, 1:J * L + 1, :, :].view()
        Sm1.shape = (num_signals, J, L, spatial_coefs, spatial_coefs)
        # precompute the fourier transform of the images
        X = np.fft.fft2(x)
        for j in range(J):
            filtersj = wavelet_filters['psi'][current_res][j].view()
            # resolution for the next layer
            resolution = max(j - oversample, 0)
            v_resolution.append(resolution)
            # fft2(| x conv Psi_j |): X is full resolution,
            # as well as the filters
            v_j = _subsample(np.fft.ifft2(_apply_fourier_mult(X, filtersj)),
                             resolution)
            V.append(v_j)
            U.append(np.abs(v_j))
            filt = wavelet_filters['phi'][resolution]
            Sm1[:, j, :, :, :] = _apply_lowpass(U[j],
                                                filt, J,
                                                spatial_coefs)
            for l in range(L):
                S_tree[(j, l)] = Sm1[:, j, l, :, :]
    # Second order scattering coeffs
    if m > 1:
        sec_order_coefs = int(J * (J - 1) * L ** 2 / 2)
        S2norder = S[:, (J * L + 1):num_coefs, :, :]  # view of the data
        S2norder.shape = (num_signals, int(sec_order_coefs/L),
                          L, spatial_coefs, spatial_coefs)
        indx = 0
        for j1 in range(J):
            # U is in the spatial domain
            Uj1 = np.fft.fft2(U[j1])
            current_res = v_resolution[j1]

            for l1 in range(Uj1.shape[1]):
                # all images single angle, all spatial coefficients
                Ujl1 = Uj1[:, l1, ]

                layer1_indexing = [(j1, l1)] * L  # for S_tree

                for j2 in range(j1 + 1, J):
                    # | U_lambda1 conv Psi_lambda2 | conv phi
                    filt = wavelet_filters['psi'][current_res][j2]
                    u_j2 = np.abs(np.fft.ifft2(
                        _apply_fourier_mult(Ujl1, filt)))
                    # computing all angles at once
                    S2norder[:, indx, :, :, :] = _apply_lowpass(
                                         u_j2,
                                         wavelet_filters['phi'][current_res],
                                         J, spatial_coefs)

                    # save tree structure
                    for l2 in range(L):
                        S_tree[((j1, l1), (j2, l2))] = \
                            S2norder[:, indx, l2, :, :]

                    indx = indx + 1
    return S, U, S_tree
