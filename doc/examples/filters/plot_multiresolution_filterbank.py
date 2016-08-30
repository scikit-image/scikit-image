"""
==========================
Multiresolution Filterbank
==========================
Here, we show how to create and access a multiresolution wavelet filter bank with Morlet filters in 2D, using the functions in filter_bank.py. We need to define the number of scales (J), angles (L), and the number of pixels (px). There are two optional parameters: the standard deviation of the low pass filter (sigma_phi) and the band pass filter (sigma_xi).
We show, for a certain scale (j) and angle (l), the same filter at different resolutions.
"""

import matplotlib.pylab as plt
import numpy as np
from skimage.filters.filter_bank import multiresolution_filter_bank_morlet2d

J = 5 #number of scales
L = 8 #number of angles
px = 128 #size of the filters
sigma_phi=0.6957 #std_0 of the low pass filter
sigma_xi=0.8506  #std_0 of the band pass filters

j=3
l=5

print('We can fix a certain j and l, and observe the filters for different resolutions')
print('j=',j,' l=',l)


#Get the multiresolution filterbank:
wavelet_bnk,littlewood_p = multiresolution_filter_bank_morlet2d(px, J=J, L=L, sigma_phi=sigma_phi, sigma_xi= sigma_xi)

num_resolutions = len(wavelet_bnk['psi'])
plt.figure(figsize=(18,6))
plt.figure(figsize=(18,6))
for r in np.arange(0,num_resolutions):
    plt.subplot(1,num_resolutions,r+1)
    f = wavelet_bnk['psi'][r][j][l,:,:]
    plt.imshow(np.abs(np.fft.fftshift(f)))
plt.show()

