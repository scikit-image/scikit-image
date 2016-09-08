"""
    Multiresolution Morlet filterbank filters
    
    Original author: Sira Ferradans, based on code developed by Ivan Dokmanic and Michael Eickenberg
"""

import numpy as np
from skimage.filters import morlet_kernel, gabor_kernel


def _ispow2(N):
    return 0 == (N & (N - 1))


def _get_filter_at_resolution(filt,j):
    """Computes the filter 'filt' at resolution 'j'
        
        Parameters
        ----------
        filt : ndarray
        Filter in the Fourier domain.
        j : int
        Resolution to be computed
        Returns
        -------
        filt_multires : ndarray
        Filter 'filt' at the resolution j, in the Fourier domain
        """
    
    cast = np.complex64
    N = filt.shape[0]  # filter is square
    
    assert _ispow2(N), 'Filter size must be an integer power of 2.'
    
    J = int(np.log2(N))
    
    # NTM: 0.5 is a cute trick for higher dimensions!
    mask = np.hstack((np.ones(int(N / 2 ** (1 + j))), 0.5, np.zeros(int(N - N / 2 ** (j + 1) - 1)))) \
        + np.hstack((np.zeros(int(N - N / 2 ** (j + 1))), 0.5, np.ones(int(N / 2 ** (1 + j) - 1))))

    mask.shape = N, 1
    
    filt_lp = filt * mask * mask.T
    if 'cast' in locals():
        filt_lp = cast(filt_lp)

    # Remember: C contiguous, last index varies "fastest" (contiguous in
    # memory) (unlike Matlab)
    fold_size = (int(2 ** j), int(N / 2 ** j), int(2 ** j), int(N / 2 ** j))
    filt_multires = filt_lp.reshape(fold_size).sum(axis=(0, 2))

    return filt_multires

def _zero_pad_filter(filter, N):
    """ Zero pad filter to (N,N)
    Zero pads 'filter' so it has a NxN size instead of its original size (n,n)

    Parameters
    ----------
    filter : (n,n) ndarray
        Filter in the Fourier domain of size (n,n)
    N : int
        Goal size of the filter, assumed to be squared (N,N)

    Returns
    -------
    padded_filter : (N,N) ndarray
        input filter 'filter' padded with zeros to the size (N,N)
    """

    if filter.shape[0] > N :
        M = np.array(filter.shape[0])
        init = np.int(np.floor(M / 2 - N / 2))
        filter = filter[init:init + N,:]

    if filter.shape[1] > N:
        M = np.array(filter.shape[1])
        init = np.int(np.floor(M / 2 - N / 2))
        filter = filter[:, init:init + N]

    left_pad = np.int64((np.array((N, N)) - np.array(filter.shape)) / 2)
    right_pad = np.int64(np.array((N, N)) - (left_pad + np.array(filter.shape)))

    padded_filter = np.lib.pad(filter, ((left_pad[0], right_pad[0]), (left_pad[1], right_pad[1])),
                               'constant', constant_values=(0, 0))
    return padded_filter


def multiresolution_filter_bank_morlet2d(N, J, L=8, sigma_phi = 0.8, sigma_xi = 0.8):
    """ Generates the multiresolution filter bank of 2D Morlet filters in the Fourier domain.
       Computes a set of 2D Morlet filters at different scales and angles, plus a low pass filter. All of them at
        at different resolutions.

    Parameters
    ----------
    N : int
        Size of the  filters, which are square (N,N)
    J : int
        Total number of scales of the filters located in the frequency domain as powers of 2.
    L : int
        Total number of angles per scale
    sigma_phi : float
        Standard deviation needed as a parameter for the low-pass filter (Gaussian), normally set to 0.8
    sigma_xi  : float
        Standard deviation needed as a parameter for every band-pass filter (Morlet), normally set to 0.8

    Returns
    -------
    Filters : Dictionary structure with the filters saved in the Fourier domain organized in the following way
            - Filters['phi'] : Low pass filter (Gaussian) in a 2D vector of size NxN
            - Filters['psi'][resolution] : Band pass filter (Morlet) saved as 4D complex array of size [J,L,N,N]
              where 'J' indexes the scale, 'L; the angles and NxN is the size of a single filter. The variable
              'resolution' goes from 0 to J.

    Examples
    --------
    Generate a multiresolution filterbank and plot all the bank pass filters at a given scale j, and angle l. They
    will be shown in the Fourier domain.

    >>>> N=128
    >>>> J=5
    >>>> L=8
    >>>> j=3
    >>>> l=5
    >>>> wavelet_bnk,littlewood_p = multiresolution_filter_bank_morlet2d(px, J=J, L=L)
    >>>> plt.figure(figsize=(18,6))
    >>>> for r in np.arange(0,num_resolutions):
            plt.subplot(1,num_resolutions,r+1)
            f = wavelet_bnk['psi'][r][j][l,:,:]
            plt.imshow(np.abs(np.fft.fftshift(f)))

     """

    wf, littlewood = filter_bank_morlet2d(N, J=J, L=L, sigma_phi=sigma_phi, sigma_xi=sigma_xi)

    multiresolution_wavelet_filters = filterbank_to_multiresolutionfilterbank(wf, J)

    return multiresolution_wavelet_filters, littlewood


def filter_bank_morlet2d(N, J=4, L=8, sigma_phi=0.8, sigma_xi=0.8):
    """ Compute a 2D complex Morlet filter bank in the Fourier domain.

    Creates a filter bank of 1+JxL number of filters in the Fourier domain, where each filter has size NxN, and differ in
    the activation frequency. All these filters are complex 2D morlet filters [1]_ , [2]_ .

    Parameters
    ----------
    N : int
        Size of the  filters, which are square (N,N)
    J : int
        Total number of scales of the filters located in the frequency domain as powers of 2.
    L : int
        Total number of angles per scale
    sigma_phi : float
        Standard deviation needed as a parameter for the low-pass filter (Gaussian), normally set to 0.8
    sigma_xi  : float
        Standard deviation needed as a parameter for every band-pass filter (Morlet), normally set to 0.8

    Returns
    -------
    Filters : Dictionary structure with the filters saved in the Fourier domain organized in the following way
            - Filters['phi'] : Low pass filter (Gaussian) in a 2D vector of size NxN
            - Filters['psi'] : Band pass filter (Morlet) saved as 4D complex array of size [J,L,N,N]
              where 'J' indexes the scale, 'L; the angles and NxN is the size of a single filter.


    littlewood_paley : (N,N) ndarray
        Measure of quality of the filter bank for image representation, it should be as close as possible to 1.
        For more information see eq. 8 in [3]_

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Filter_bank
    .. [2] https://en.wikipedia.org/wiki/Discrete_wavelet_transform
    .. [3] Bruna, J., Mallat, S. 'Invariant Scattering Convolutional Networks'. IEEE Transactions on PAMI, 2012.


    Examples
    --------

    Generate a filter bank and show all the band pass filters in the Fourier domain.

    >>>>J = 3
    >>>>L=8
    >>>>px = 32
    >>>>wavelet_filters, littlewood = filter_bank_morlet2d(px, J=J, L=L, sigma_phi=0.6957,sigma_xi=0.8506 )
    >>>>print('params consistent? J2=',J2,' L2=',L2,' N2=', N2)
        print('Show low pass filter in the Fourier domain')
        plt.imshow(np.abs(np.fft.fftshift(filters['phi'])))
        plt.show()
        print('.. and the band pass filters in the Fourier domain:')

        for j in np.arange(0,J):
            print('j=',j)
            plt.figure(figsize=(18,6))
            for l in np.arange(0,L):
                plt.subplot(1, L, l+1)
                plt.imshow(np.abs(np.fft.fftshift(filters['psi'][j][l, :, :])))

            plt.show()

    """
    max_scale = 2 ** (float(J - 1))

    sigma = sigma_phi * max_scale
    freq = 0.

    filter_phi = np.ndarray((N,N), dtype='complex')
    littlewood_paley = np.zeros((N, N), dtype='single')

    # Low pass
    filter_phi = np.fft.fft2(np.fft.fftshift(_zero_pad_filter(gabor_kernel(freq, theta=0., sigma_x=sigma, sigma_y=sigma),N)))

    ## Band pass filters:
    ## psi: Create band-pass filters
    # constant values for psi
    xi_psi = 3. /4 * np.pi
    slant = 4. / L

    filters_psi = []

    for j, scale in enumerate(2. ** np.arange(J)):
        angles = np.zeros((L, N, N), dtype='complex')
        for l, theta in enumerate(np.pi * np.arange(L) / float(L)):
            sigma = sigma_xi * scale
            xi = xi_psi / scale

            sigma_x = sigma
            sigma_y = sigma / slant
            freq = xi / (np.pi * 2)

            psi = morlet_kernel(freq, theta=theta, sigma_x=sigma_x, sigma_y=sigma_y,n_stds=12)
      
            #needs a small shift for odd sizes
            if (psi.shape[0] % 2 > 0):
                if (psi.shape[1] % 2 > 0):
                    Psi = _zero_pad_filter(psi[:-1, :-1], N)
                else:
                    Psi = _zero_pad_filter(psi[:-1, :], N)
            else:
                if (psi.shape[1] % 2 > 0):
                    Psi = _zero_pad_filter(psi[:, :-1], N)
                else:
                    Psi = _zero_pad_filter(psi, N)

            angles[l, :, :] = np.fft.fft2(np.fft.fftshift(0.5*Psi))

        littlewood_paley += np.sum(np.abs(angles) ** 2, axis=0)
        filters_psi.append(angles)

    lwp_max = littlewood_paley.max()

    for filt in filters_psi:
        filt /= np.sqrt(lwp_max/2)

    Filters = dict(phi=filter_phi, psi=filters_psi)

    return Filters, littlewood_paley*2


def filterbank_to_multiresolutionfilterbank(filters, max_resolution):
    """ Converts a filter bank into a multiresolution filterbank
    For every filter of a filter bank, compute this same filter at different resolutions (differnt sizes). The filters
    are assumed to be in the Fourier domain, as well as the output multiresolution filters.

    Parameters
    ----------
    filters : dictionary
        Set of filters stored in a dictionary in the following way:
         - filters['phi'] : Low pass filter (Gaussian) in a 2D vector of size NxN
         - filters['psi'] : Band pass filter (Morlet) saved as 4D complex array of size [J,L,N,N]
              where 'J' indexes the scale, 'L; the angles and NxN is the size of a single filter.

    max_resolution : int
        number of resolutions to compute for every filter (thus for every scale and angle)

    Returns
    -------
    filters_multires : dictionary
        Set of filters in the Fourier domain, at different scales, angles and resolutions.
        See multiresolution_filter_bank_morlet2d for more details on the Filters_multires structure.

    """
    J = len(filters['psi']) #scales
    L = len(filters['psi'][0]) #angles
    N = filters['psi'][0].shape[-1] #size at max scale

    Phi_multires = []
    Psi_multires = []
    for res in np.arange(0,max_resolution):
        Phi_multires.append(_get_filter_at_resolution(filters['phi'],res))

        aux_filt_psi = np.ndarray((J,L,int(N/2**res),int(N/2**res)), dtype='complex64')
        for j in np.arange(0,J):
            for l in np.arange(0,L):
                aux_filt_psi[j,l,:,:] = _get_filter_at_resolution(filters['psi'][j][l,:,:],res)

        Psi_multires.append(aux_filt_psi)

    filters_multires = dict(phi=Phi_multires, psi=Psi_multires)
    return filters_multires







