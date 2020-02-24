"""
========================================
Calibrating Denoisers Using J-Invariance
========================================

In this example, we show how to find an optimally calibrated
version of any denoising algorithm.

The calibration method is based on the `noise2self` algorithm of [1]_.

.. [1] J. Batson & L. Royer. Noise2Self: Blind Denoising by Self-Supervision,
       International Conference on Machine Learning, p. 524-533 (2019).

.. seealso::
   More details about the method are given in the full tutorial
   :ref:`sphx_glr_auto_examples_filters_plot_j_invariant_tutorial.py`.
"""

#####################################################################
# Calibrating a wavelet denoiser

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

from skimage.data import chelsea, hubble_deep_field
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.restoration import (calibrate_denoiser,
                                 denoise_wavelet,
                                 denoise_tv_chambolle, denoise_nl_means,
                                 estimate_sigma)
from skimage.util import img_as_float, random_noise
from skimage.color import rgb2gray


image = img_as_float(chelsea())
sigma = 0.3
noisy = random_noise(image, var=sigma ** 2)

# Parameters to test when calibrating the denoising algorithm
sigma_range = np.arange(0.05, 0.5, 0.05)

parameter_ranges = {'sigma': np.arange(0.1, 0.3, 0.02),
                    'wavelet': ['db1', 'db2'],
                    'convert2ycbcr': [True, False],
                    'multichannel': [True]}

# Denoised image using default parameters of `denoise_wavelet`
default_output = denoise_wavelet(noisy, multichannel=True)

# Calibrate denoiser
calibrated_denoiser = calibrate_denoiser(noisy,
                                         denoise_wavelet,
                                         denoise_parameters=parameter_ranges)

# Denoised image using calibrated denoiser
calibrated_output = calibrated_denoiser(noisy)

fig, axes = plt.subplots(1, 3, sharey=True, figsize=(15, 5))

for ax, img, title in zip(axes,
                          [noisy, default_output, calibrated_output],
                          ['Noisy Image', 'Denoised (Default)',
                           'Denoised (Calibrated)']):
    ax.imshow(img)
    ax.set_title(title)
    ax.set_yticks([])
    ax.set_xticks([])

plt.show()
