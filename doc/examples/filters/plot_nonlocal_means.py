"""
=================================================
Non-local means denoising for preserving textures
=================================================

In this example, we denoise a detail of the astronaut image using the non-local
means filter. The non-local means algorithm replaces the value of a pixel by an
average of a selection of other pixels values: small patches centered on the
other pixels are compared to the patch centered on the pixel of interest, and
the average is performed only for pixels that have patches close to the current
patch. As a result, this algorithm can restore well textures, that would be
blurred by other denoising algoritm.
"""
import numpy as np
import matplotlib.pyplot as plt

from skimage import data, img_as_float
from skimage.restoration import denoise_nl_means


astro = img_as_float(data.astronaut())
astro = astro[30:180, 150:300]

noisy = astro + 0.3 * np.random.random(astro.shape)
noisy = np.clip(noisy, 0, 1)

denoise = denoise_nl_means(noisy, 7, 9, 0.08, multichannel=True)

# slow algorithm
denoise = denoise_nl_means(noisy, h=1.15 * sigma_est, fast_mode=False,
                           **patch_kw)

# slow algorithm, sigma provided
denoise2 = denoise_nl_means(noisy, h=0.8 * sigma_est, sigma=sigma_est,
                            fast_mode=False, **patch_kw)

# fast algorithm
denoise_fast = denoise_nl_means(noisy, h=0.8 * sigma_est, fast_mode=True,
                                **patch_kw)

# fast algorithm, sigma provided
denoise2_fast = denoise_nl_means(noisy, h=0.6 * sigma_est, sigma=sigma_est,
                                 fast_mode=True, **patch_kw)

fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(8, 6),
                       sharex=True, sharey=True)

ax[0].imshow(noisy)
ax[0].axis('off')
ax[0].set_title('noisy')
ax[1].imshow(denoise)
ax[1].axis('off')
ax[1].set_title('non-local means')

fig.tight_layout()

plt.show()
